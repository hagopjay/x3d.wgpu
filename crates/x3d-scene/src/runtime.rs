//! The live part of a scene: what X3D's event system changes over time.
//!
//! 🎨 **How X3D animates.** Nothing moves by itself. A `TimeSensor` emits
//! `fraction_changed` (0 → 1 each cycle); a `ROUTE` carries that number
//! into an interpolator's `set_fraction`; the interpolator emits a value
//! (a position, a colour, a rotation…) and another ROUTE carries it into a
//! node's field, such as a Transform's `set_rotation`. One frame of
//! animation is one *cascade* of these events, and X3D's rule is that each
//! ROUTE fires at most once per timestamp, so loops can't run forever.
//!
//! 🦀 **Ownership trick.** [`Scene::advance`] needs to read the event graph
//! *and* write the scene it belongs to. Borrowing both from `self` at once
//! won't compile, so it moves the graph out with `std::mem::take`, works,
//! then puts it back. No `RefCell`, no `unsafe`, no clones.

use std::collections::{HashMap, VecDeque};

use glam::{Mat4, Quat, Vec2, Vec3};

use crate::scene::{LightKind, Scene};

/// Seconds between the Unix epoch and "now" when the scene loads, as far as
/// TimeSensors are concerned. A huge value makes `startTime='0'` mean "long
/// ago" like in a browser: looping sensors run, one-shot ones have finished.
pub const LOAD_TIME: f64 = 1.0e9;

/// One event's payload.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Value {
    Bool(bool),
    Int(i32),
    Float(f32),
    Time(f64),
    Vec2(Vec2),
    Vec3(Vec3),
    Rotation(Quat),
}

impl Value {
    fn float(self) -> Option<f32> {
        match self {
            Self::Float(f) => Some(f),
            Self::Time(t) => Some(t as f32),
            Self::Int(i) => Some(i as f32),
            _ => None,
        }
    }
    fn vec3(self) -> Option<Vec3> {
        match self {
            Self::Vec3(v) => Some(v),
            Self::Vec2(v) => Some(v.extend(0.0)),
            _ => None,
        }
    }
    fn vec2(self) -> Option<Vec2> {
        match self {
            Self::Vec2(v) => Some(v),
            Self::Vec3(v) => Some(v.truncate()),
            _ => None,
        }
    }
    fn rotation(self) -> Option<Quat> {
        match self {
            Self::Rotation(q) => Some(q),
            _ => None,
        }
    }
    fn boolean(self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(b),
            _ => None,
        }
    }
    fn int(self) -> Option<i32> {
        match self {
            Self::Int(i) => Some(i),
            Self::Float(f) => Some(f as i32),
            _ => None,
        }
    }
    fn time(self) -> Option<f64> {
        match self {
            Self::Time(t) => Some(t),
            Self::Float(f) => Some(f64::from(f)),
            _ => None,
        }
    }
}

/// `set_translation`, `translation_changed` and `translation` all name the
/// same field. Routes are matched on this normal form.
pub fn field_name(f: &str) -> &str {
    let f = f.strip_prefix("set_").unwrap_or(f);
    f.strip_suffix("_changed").unwrap_or(f)
}

// ---------------------------------------------------------------------------
// The transform tree
// ---------------------------------------------------------------------------

/// A Transform's fields, kept separately so events can change one of them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Trs {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
    pub scale_orientation: Quat,
    pub center: Vec3,
}

impl Default for Trs {
    fn default() -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            scale_orientation: Quat::IDENTITY,
            center: Vec3::ZERO,
        }
    }
}

impl Trs {
    /// X3D's Transform: `T · C · R · SR · S · SR⁻¹ · C⁻¹`.
    pub fn matrix(&self) -> Mat4 {
        Mat4::from_translation(self.translation + self.center)
            * Mat4::from_quat(self.rotation)
            * Mat4::from_quat(self.scale_orientation)
            * Mat4::from_scale(self.scale)
            * Mat4::from_quat(self.scale_orientation.inverse())
            * Mat4::from_translation(-self.center)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Local {
    /// Never changes (the scene root, an Inline's placement).
    Fixed(Mat4),
    /// A Transform; events may change its fields.
    Trs(Trs),
}

/// One node of the transform tree. Parents always come before children in
/// [`Scene::nodes`], so one forward pass updates every `world`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransformNode {
    pub parent: Option<u32>,
    pub local: Local,
    pub world: Mat4,
}

impl TransformNode {
    pub fn root() -> Self {
        Self {
            parent: None,
            local: Local::Fixed(Mat4::IDENTITY),
            world: Mat4::IDENTITY,
        }
    }
    pub fn local_matrix(&self) -> Mat4 {
        match self.local {
            Local::Fixed(m) => m,
            Local::Trs(t) => t.matrix(),
        }
    }
}

/// A 2D TextureTransform's fields: `Tc' = −C × S × R × C × T × Tc`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextureTransform2D {
    pub center: Vec2,
    pub rotation: f32,
    pub scale: Vec2,
    pub translation: Vec2,
}

impl TextureTransform2D {
    pub fn matrix(&self) -> Mat4 {
        Mat4::from_translation(-self.center.extend(0.0))
            * Mat4::from_scale(self.scale.extend(1.0))
            * Mat4::from_rotation_z(self.rotation)
            * Mat4::from_translation(self.center.extend(0.0))
            * Mat4::from_translation(self.translation.extend(0.0))
    }
}

// ---------------------------------------------------------------------------
// The event graph
// ---------------------------------------------------------------------------

/// Keyed values of an interpolator, one variant per X3D type.
#[derive(Debug, Clone, PartialEq)]
pub enum KeyValues {
    Float(Vec<f32>),
    Vec2(Vec<Vec2>),
    Vec3(Vec<Vec3>),
    /// ColorInterpolator: X3D interpolates colours in HSV space.
    Color(Vec<Vec3>),
    Rotation(Vec<Quat>),
    Int(Vec<i32>),
    Bool(Vec<bool>),
}

/// Nodes that *process* events.
#[derive(Debug, Clone, PartialEq)]
pub enum Behaviour {
    TimeSensor {
        cycle: f64,
        looping: bool,
        start: f64,
        stop: f64,
        enabled: bool,
        active: bool,
    },
    /// Scalar/Position/Position2D/Color/Orientation interpolators, and the
    /// Integer/Boolean sequencers (which step instead of blending).
    Interpolator {
        key: Vec<f32>,
        values: KeyValues,
    },
    BooleanFilter,
    BooleanToggle {
        state: bool,
    },
}

/// Nodes whose fields events *change*, with the scene parts they own.
/// A DEF'd node USE'd twice owns two of each.
#[derive(Debug, Clone, PartialEq)]
pub enum Target {
    /// Indices into [`Scene::nodes`].
    Transform(Vec<u32>),
    /// Indices into [`Scene::materials`].
    Material(Vec<u32>),
    /// Indices into [`Scene::lights`].
    Light(Vec<u32>),
    /// Indices into [`Scene::switches`].
    Switch(Vec<u32>),
    /// Indices into [`Scene::materials`] whose texture matrix this sets.
    TextureTransform {
        materials: Vec<u32>,
        fields: TextureTransform2D,
    },
    /// A ComposedShader field: index into `Scene::shader_fields`.
    ShaderField(Vec<u32>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum AnimNode {
    Behaviour(Behaviour),
    Target(Target),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Route {
    pub from: u32,
    /// Normal form (see [`field_name`]).
    pub from_field: String,
    pub to: u32,
    pub to_field: String,
}

/// The event graph of a scene.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Animation {
    pub nodes: Vec<AnimNode>,
    /// DEF names, for messages.
    pub names: Vec<String>,
    pub routes: Vec<Route>,
}

impl Animation {
    pub fn is_empty(&self) -> bool {
        self.routes.is_empty()
    }

    /// Append another scene's graph, renumbering everything it refers to.
    pub(crate) fn absorb(&mut self, other: &Animation, offsets: &Offsets) {
        let base = self.nodes.len() as u32;
        let shift = |v: &[u32], by: u32| v.iter().map(|i| i + by).collect::<Vec<_>>();
        for n in &other.nodes {
            self.nodes.push(match n {
                AnimNode::Behaviour(b) => AnimNode::Behaviour(b.clone()),
                AnimNode::Target(t) => AnimNode::Target(match t {
                    Target::Transform(v) => Target::Transform(shift(v, offsets.nodes)),
                    Target::Material(v) => Target::Material(shift(v, offsets.materials)),
                    Target::Light(v) => Target::Light(shift(v, offsets.lights)),
                    Target::Switch(v) => Target::Switch(shift(v, offsets.switches)),
                    Target::TextureTransform { materials, fields } => Target::TextureTransform {
                        materials: shift(materials, offsets.materials),
                        fields: *fields,
                    },
                    Target::ShaderField(v) => Target::ShaderField(shift(v, offsets.shader_fields)),
                }),
            });
        }
        self.names.extend(other.names.iter().cloned());
        self.routes.extend(other.routes.iter().map(|r| Route {
            from: r.from + base,
            to: r.to + base,
            ..r.clone()
        }));
    }
}

/// How far to shift each index when one scene is spliced into another.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Offsets {
    pub nodes: u32,
    pub materials: u32,
    pub lights: u32,
    pub switches: u32,
    pub shader_fields: u32,
}

// ---------------------------------------------------------------------------
// Interpolation
// ---------------------------------------------------------------------------

/// Which keys bracket `f`, and how far between them (0..1).
fn bracket(key: &[f32], f: f32) -> (usize, usize, f32) {
    let last = key.len().saturating_sub(1);
    if key.is_empty() || f <= key[0] {
        return (0, 0, 0.0);
    }
    if f >= key[last] {
        return (last, last, 0.0);
    }
    // 🦀 `partition_point` = binary search for "first key > f".
    let hi = key.partition_point(|&k| k <= f);
    let lo = hi - 1;
    let span = key[hi] - key[lo];
    (
        lo,
        hi,
        if span > 0.0 {
            (f - key[lo]) / span
        } else {
            0.0
        },
    )
}

/// The step a sequencer is on: the last key ≤ f.
fn step(key: &[f32], f: f32) -> usize {
    key.partition_point(|&k| k <= f).saturating_sub(1)
}

fn rgb_to_hsv(c: Vec3) -> Vec3 {
    let (max, min) = (c.max_element(), c.min_element());
    let d = max - min;
    let h = if d <= 0.0 {
        0.0
    } else if max == c.x {
        ((c.y - c.z) / d).rem_euclid(6.0)
    } else if max == c.y {
        (c.z - c.x) / d + 2.0
    } else {
        (c.x - c.y) / d + 4.0
    } / 6.0;
    let s = if max > 0.0 { d / max } else { 0.0 };
    Vec3::new(h, s, max)
}

fn hsv_to_rgb(hsv: Vec3) -> Vec3 {
    let (h, s, v) = (hsv.x.rem_euclid(1.0) * 6.0, hsv.y, hsv.z);
    let c = v * s;
    let x = c * (1.0 - ((h % 2.0) - 1.0).abs());
    let (r, g, b) = match h as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    Vec3::new(r, g, b) + Vec3::splat(v - c)
}

/// Blend two colours the X3D way: through HSV, the short way round the hue circle.
pub fn lerp_color(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    let (mut ha, hb) = (rgb_to_hsv(a), rgb_to_hsv(b));
    if (hb.x - ha.x).abs() > 0.5 {
        ha.x += if hb.x > ha.x { 1.0 } else { -1.0 };
    }
    hsv_to_rgb(ha.lerp(hb, t))
}

fn interpolate(key: &[f32], values: &KeyValues, f: f32) -> Option<Value> {
    let (lo, hi, t) = bracket(key, f);
    Some(match values {
        KeyValues::Float(v) => Value::Float(lerp(*v.get(lo)?, *v.get(hi)?, t)),
        KeyValues::Vec2(v) => Value::Vec2(v.get(lo)?.lerp(*v.get(hi)?, t)),
        KeyValues::Vec3(v) => Value::Vec3(v.get(lo)?.lerp(*v.get(hi)?, t)),
        KeyValues::Color(v) => Value::Vec3(lerp_color(*v.get(lo)?, *v.get(hi)?, t)),
        KeyValues::Rotation(v) => Value::Rotation(v.get(lo)?.slerp(*v.get(hi)?, t)),
        KeyValues::Int(v) => Value::Int(*v.get(step(key, f))?),
        KeyValues::Bool(v) => Value::Bool(*v.get(step(key, f))?),
    })
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

// ---------------------------------------------------------------------------
// One frame
// ---------------------------------------------------------------------------

/// What one call to [`Scene::advance`] changed, so a renderer can update
/// only what it must.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Changes {
    pub transforms: bool,
    pub materials: bool,
    pub lights: bool,
    pub switches: bool,
    pub shader_fields: bool,
    /// Events delivered this frame.
    pub events: usize,
}

impl Changes {
    pub fn any(&self) -> bool {
        self.transforms || self.materials || self.lights || self.switches || self.shader_fields
    }
}

impl Scene {
    /// Run the event graph at `t` seconds after loading and apply what it
    /// changes. Deterministic: the same `t` always gives the same state for
    /// time-driven animation, so `render --time 2.5` is reproducible.
    pub fn advance(&mut self, t: f64) -> Changes {
        let mut anim = std::mem::take(&mut self.animation);
        let changes = self.cascade(&mut anim, LOAD_TIME + t.max(0.0));
        self.animation = anim;
        changes
    }

    fn cascade(&mut self, anim: &mut Animation, now: f64) -> Changes {
        let mut changes = Changes::default();
        // Routes by (source node, source field).
        let mut by_source: HashMap<(u32, &str), Vec<usize>> = HashMap::new();
        for (i, r) in anim.routes.iter().enumerate() {
            by_source
                .entry((r.from, r.from_field.as_str()))
                .or_default()
                .push(i);
        }
        let mut fired = vec![false; anim.routes.len()];
        let mut queue: VecDeque<(u32, &'static str, Value)> = VecDeque::new();

        // 1. Clocks tick.
        for (i, node) in anim.nodes.iter_mut().enumerate() {
            if let AnimNode::Behaviour(Behaviour::TimeSensor {
                cycle,
                looping,
                start,
                stop,
                enabled,
                active,
            }) = node
            {
                let i = i as u32;
                let cycle = cycle.max(1e-6);
                let begun = start.max(LOAD_TIME - 1e-9);
                let running = *enabled
                    && now >= *start
                    && (*stop <= *start || now < *stop)
                    && (*looping || now < *start + cycle);
                if running {
                    let elapsed = now - begun;
                    let fraction = if *looping {
                        (elapsed / cycle).fract()
                    } else {
                        ((now - *start) / cycle).min(1.0)
                    };
                    queue.push_back((i, "fraction", Value::Float(fraction as f32)));
                    queue.push_back((i, "elapsedTime", Value::Time(elapsed)));
                    queue.push_back((i, "time", Value::Time(now)));
                } else if *active {
                    // A one-shot cycle just ended: X3D sends fraction 1 last.
                    queue.push_back((i, "fraction", Value::Float(1.0)));
                }
                if running != *active {
                    *active = running;
                    queue.push_back((i, "isActive", Value::Bool(running)));
                }
            }
        }

        // 2. Events flow along routes until nothing new happens.
        let mut writes: Vec<(u32, String, Value)> = Vec::new();
        while let Some((from, field, value)) = queue.pop_front() {
            let Some(routes) = by_source.get(&(from, field)) else {
                continue;
            };
            for &ri in routes {
                if std::mem::replace(&mut fired[ri], true) {
                    continue; // each route fires once per timestamp
                }
                changes.events += 1;
                let route = &anim.routes[ri];
                let to = route.to;
                match &mut anim.nodes[to as usize] {
                    AnimNode::Behaviour(b) => {
                        for (out, v) in receive(b, &route.to_field, value) {
                            queue.push_back((to, out, v));
                        }
                    }
                    AnimNode::Target(_) => writes.push((to, route.to_field.clone(), value)),
                }
            }
        }

        // 3. Targets apply what reached them.
        for (to, field, value) in writes {
            if let AnimNode::Target(target) = &mut anim.nodes[to as usize] {
                self.apply(target, &field, value, &mut changes);
            }
        }
        if changes.transforms {
            self.update_worlds();
        }
        if changes.transforms || changes.lights {
            self.update_light_worlds();
        }
        changes
    }

    fn apply(&mut self, target: &mut Target, field: &str, v: Value, changes: &mut Changes) {
        match target {
            Target::Transform(nodes) => {
                for &n in nodes.iter() {
                    let node = &mut self.nodes[n as usize];
                    let Local::Trs(trs) = &mut node.local else {
                        continue;
                    };
                    let ok = match field {
                        "translation" => v.vec3().map(|x| trs.translation = x),
                        "rotation" => v.rotation().map(|x| trs.rotation = x),
                        "scale" => v.vec3().map(|x| trs.scale = x),
                        "center" => v.vec3().map(|x| trs.center = x),
                        "scaleOrientation" => v.rotation().map(|x| trs.scale_orientation = x),
                        _ => None,
                    };
                    changes.transforms |= ok.is_some();
                }
            }
            Target::Material(ids) => {
                for &id in ids.iter() {
                    let m = &mut self.materials[id as usize];
                    let ok = match field {
                        "emissiveColor" => v.vec3().map(|c| {
                            m.emissive = c;
                            if m.unlit {
                                m.diffuse = c;
                            }
                        }),
                        "diffuseColor" | "baseColor" => v.vec3().map(|c| m.diffuse = c),
                        "specularColor" => v.vec3().map(|c| m.specular = c),
                        "transparency" => v.float().map(|x| m.transparency = x.clamp(0.0, 1.0)),
                        "shininess" => v.float().map(|x| m.shininess = x),
                        "ambientIntensity" => v.float().map(|x| m.ambient_intensity = x),
                        _ => None,
                    };
                    changes.materials |= ok.is_some();
                }
                if changes.materials {
                    self.sync_draw_materials(ids);
                }
            }
            Target::Light(ids) => {
                for &id in ids.iter() {
                    let l = &mut self.lights[id as usize];
                    let ok = match field {
                        "color" => v.vec3().map(|c| l.color = c),
                        "intensity" => v.float().map(|x| l.intensity = x),
                        "ambientIntensity" => v.float().map(|x| l.ambient_intensity = x),
                        "on" => v.boolean().map(|b| l.on = b),
                        "location" => v.vec3().map(|p| match &mut l.local {
                            LightKind::Point { location, .. }
                            | LightKind::Spot { location, .. } => *location = p,
                            LightKind::Directional { .. } => {}
                        }),
                        "direction" => v.vec3().map(|d| match &mut l.local {
                            LightKind::Directional { direction }
                            | LightKind::Spot { direction, .. } => *direction = d,
                            LightKind::Point { .. } => {}
                        }),
                        _ => None,
                    };
                    changes.lights |= ok.is_some();
                }
            }
            Target::Switch(ids) => {
                if let ("whichChoice", Some(choice)) = (field, v.int()) {
                    for &id in ids.iter() {
                        self.switches[id as usize] = choice;
                    }
                    changes.switches = true;
                }
            }
            Target::TextureTransform { materials, fields } => {
                let ok = match field {
                    "translation" => v.vec2().map(|x| fields.translation = x),
                    "scale" => v.vec2().map(|x| fields.scale = x),
                    "center" => v.vec2().map(|x| fields.center = x),
                    "rotation" => v.float().map(|x| fields.rotation = x),
                    _ => None,
                };
                if ok.is_some() {
                    let m = fields.matrix();
                    for &id in materials.iter() {
                        self.materials[id as usize].texture_matrix = m;
                    }
                    self.sync_draw_materials(materials);
                    changes.materials = true;
                }
            }
            Target::ShaderField(ids) => {
                for &id in ids.iter() {
                    self.shader_fields[id as usize] = v;
                }
                changes.shader_fields = true;
            }
        }
    }

    /// Copy changed materials into the draws that use them, so `draw.material`
    /// is always the live value.
    fn sync_draw_materials(&mut self, ids: &[u32]) {
        for d in &mut self.draws {
            if ids.contains(&d.material_id) {
                d.material = self.materials[d.material_id as usize].clone();
            }
        }
    }

    /// Recompute every node's world matrix (parents first), then the draws'.
    pub fn update_worlds(&mut self) {
        for i in 0..self.nodes.len() {
            let local = self.nodes[i].local_matrix();
            let parent = self.nodes[i]
                .parent
                .map_or(Mat4::IDENTITY, |p| self.nodes[p as usize].world);
            self.nodes[i].world = parent * local;
        }
        for d in &mut self.draws {
            d.transform = self.nodes[d.node as usize].world;
        }
    }

    /// Lights are stored in their node's space; place them in the world.
    pub fn update_light_worlds(&mut self) {
        for l in &mut self.lights {
            let m = self.nodes[l.node as usize].world;
            l.kind = l.local_in(&m);
        }
    }

    /// Is this draw on the chosen side of every Switch it sits under?
    pub fn switched_on(&self, guards: &[crate::scene::SwitchGuard]) -> bool {
        guards
            .iter()
            .all(|g| self.switches[g.switch as usize] == g.child as i32)
    }
}

/// A behaviour receives one event and answers with zero or more events.
fn receive(b: &mut Behaviour, field: &str, v: Value) -> Vec<(&'static str, Value)> {
    match b {
        Behaviour::Interpolator { key, values } => match (field, v.float()) {
            ("fraction", Some(f)) => interpolate(key, values, f)
                .map(|v| ("value", v))
                .into_iter()
                .collect(),
            _ => Vec::new(),
        },
        Behaviour::BooleanFilter => match (field, v.boolean()) {
            ("boolean", Some(x)) => {
                let mut out = vec![("inputNegate", Value::Bool(!x))];
                out.push(if x {
                    ("inputTrue", Value::Bool(true))
                } else {
                    ("inputFalse", Value::Bool(false))
                });
                out
            }
            _ => Vec::new(),
        },
        Behaviour::BooleanToggle { state } => match (field, v.boolean()) {
            ("boolean", Some(true)) => {
                *state = !*state;
                vec![("toggle", Value::Bool(*state))]
            }
            _ => Vec::new(),
        },
        Behaviour::TimeSensor {
            start,
            stop,
            enabled,
            cycle,
            looping,
            ..
        } => {
            match field {
                "startTime" => {
                    if let Some(t) = v.time() {
                        *start = t;
                    }
                }
                "stopTime" => {
                    if let Some(t) = v.time() {
                        *stop = t;
                    }
                }
                "enabled" => {
                    if let Some(x) = v.boolean() {
                        *enabled = x;
                    }
                }
                "cycleInterval" => {
                    if let Some(t) = v.time() {
                        *cycle = t;
                    }
                }
                "loop" => {
                    if let Some(x) = v.boolean() {
                        *looping = x;
                    }
                }
                _ => {}
            }
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_names_normalise() {
        assert_eq!(field_name("set_fraction"), "fraction");
        assert_eq!(field_name("value_changed"), "value");
        assert_eq!(field_name("elapsedTime"), "elapsedTime");
    }

    #[test]
    fn bracket_and_step() {
        let key = [0.0, 0.5, 1.0];
        assert_eq!(bracket(&key, 0.25), (0, 1, 0.5));
        assert_eq!(bracket(&key, 2.0), (2, 2, 0.0));
        assert_eq!(step(&key, 0.6), 1);
    }

    #[test]
    fn colour_interpolation_goes_through_hsv() {
        // Red → blue through HSV passes magenta, not the dark purple of RGB.
        let mid = lerp_color(Vec3::X, Vec3::Z, 0.5);
        assert!((mid - Vec3::new(1.0, 0.0, 1.0)).length() < 1e-5, "{mid}");
        assert!((lerp_color(Vec3::X, Vec3::Z, 0.0) - Vec3::X).length() < 1e-6);
    }

    #[test]
    fn hsv_round_trip() {
        for c in [
            Vec3::new(0.2, 0.5, 0.9),
            Vec3::new(0.9, 0.1, 0.1),
            Vec3::splat(0.4),
        ] {
            assert!((hsv_to_rgb(rgb_to_hsv(c)) - c).length() < 1e-5);
        }
    }
}
