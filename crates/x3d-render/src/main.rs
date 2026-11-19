//! `x3d-view`: look at X3D files on the GPU.
//!
//! ```text
//! x3d-view view   scenes/still_life.x3d            # interactive window
//! x3d-view render scenes/still_life.x3d -o out.png # headless PNG
//! x3d-view render scenes/still_life.x3d --sheet -o sheet.png   # all 6 debug views
//! x3d-view info   scenes/still_life.x3d            # what was kept / skipped
//! x3d-view adapters                                # which GPUs wgpu can see
//! ```
//!
//! 🦀 `clap`'s derive API: the CLI *is* these structs. Doc comments become
//! `--help` text, field types become parsers, enums become subcommands.

use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use x3d_render::cull::Cull;
use x3d_render::renderer::{Encode, IblModel};
use x3d_render::{Backend, ColorMode, DebugView, Gpu, Profile, RenderSettings, Shot, ToneMap};
use x3d_scene::LoadOptions;

#[derive(Parser, Debug)]
#[command(name = "x3d-view", version, about = "A wgpu (Vulkan) renderer for X3D scenes")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Open an interactive window.
    #[cfg(feature = "window")]
    View {
        file: PathBuf,
        /// Load the WGSL shader from this file and hot-reload it on save.
        #[arg(long)]
        shader: Option<PathBuf>,
        /// Save a PNG here after --exit-after frames, then quit (for scripts/CI).
        #[arg(long, requires = "exit_after")]
        screenshot: Option<PathBuf>,
        #[arg(long)]
        exit_after: Option<u32>,
        #[command(flatten)]
        common: Common,
    },
    /// Render to a PNG without a window.
    Render {
        file: PathBuf,
        #[arg(short, long, default_value = "out.png")]
        output: PathBuf,
        /// Which <Viewpoint> (0-based). Default: the first, or frame the scene.
        #[arg(long)]
        viewpoint: Option<usize>,
        /// A contact sheet of every debug view instead of one image.
        #[arg(long)]
        sheet: bool,
        /// Seconds after loading: TimeSensors and ROUTEs run up to this moment.
        #[arg(long, default_value_t = 0.0)]
        time: f64,
        #[command(flatten)]
        common: Common,
    },
    /// Render N frames off-screen and print per-pass GPU times (timestamp
    /// queries), CPU time, culling counts and pipeline-cache effect.
    Bench {
        file: PathBuf,
        #[arg(long, default_value_t = 60)]
        frames: usize,
        #[arg(long)]
        viewpoint: Option<usize>,
        #[arg(long, default_value_t = 0.0)]
        time: f64,
        #[command(flatten)]
        common: Common,
    },
    /// Render conformance/expected.json and write a receipt (every number
    /// names its source: the measurement, the validator, the adapter).
    Conform {
        /// The conformance folder (expected.json, validators/, scenes/).
        #[arg(default_value = "conformance")]
        dir: PathBuf,
        /// Write the JSON receipt here.
        #[arg(long)]
        receipt: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = Backend::Auto)]
        backend: Backend,
    },
    /// Print what the loader kept, approximated and skipped.
    Info {
        file: PathBuf,
        /// List every shape, too.
        #[arg(long)]
        shapes: bool,
        /// Also load Inlines marked load='false'.
        #[arg(long)]
        load_all_inlines: bool,
    },
    /// Translate and compile every ComposedShader in a file; show what fails and why.
    Shaders {
        file: PathBuf,
        /// Print the translated GLSL 4.50 too.
        #[arg(long)]
        show: bool,
    },
    /// List the GPU adapters wgpu can see.
    Adapters {
        #[arg(long, value_enum, default_value_t = Backend::Auto)]
        backend: Backend,
        /// Also open the best adapter and print the budget table: every
        /// setting this GPU gets and the limit/feature it came from.
        #[arg(long)]
        budget: bool,
        #[arg(long, value_enum, default_value_t = Profile::Auto)]
        profile: Profile,
    },
}

/// Options shared by `view` and `render`.
/// 🦀 `#[command(flatten)]` splices these fields into each subcommand.
#[derive(Args, Debug)]
struct Common {
    #[arg(long, value_enum, default_value_t = Backend::Auto)]
    backend: Backend,
    /// `weak` emulates a WebGL2-class GPU (downlevel limits, no optional features).
    #[arg(long, value_enum, default_value_t = Profile::Auto)]
    profile: Profile,
    /// Image or window size, e.g. 1920x1080.
    #[arg(long, default_value = "1280x720")]
    size: Size,
    /// Multisample anti-aliasing: 1 (off) or 4.
    #[arg(long, default_value_t = 4)]
    msaa: u32,
    #[arg(long, value_enum, default_value_t = DebugView::Lit)]
    view: DebugView,
    #[arg(long, value_enum, default_value_t = ColorMode::Display)]
    color_mode: ColorMode,
    #[arg(long)]
    wireframe: bool,
    /// Force the headlight on or off (default: what the scene's NavigationInfo says).
    #[arg(long)]
    headlight: Option<bool>,
    /// Don't cull back faces, even of solid geometry.
    #[arg(long)]
    no_cull: bool,
    /// Segments around spheres, cylinders and cones.
    #[arg(long, default_value_t = 48)]
    segments: u32,
    /// Fail on the first bad node instead of skipping it.
    #[arg(long)]
    strict: bool,
    /// Also load Inlines marked load='false'.
    #[arg(long)]
    load_all_inlines: bool,
    /// Brighten or darken PhysicalMaterial lighting (1 = as X_ITE draws it);
    /// with --tone-map or --bloom, all light.
    #[arg(long, default_value_t = 1.0)]
    exposure: f32,
    /// Multiply line widths and point sizes (e.g. 2 on a HiDPI screen).
    #[arg(long, default_value_t = 1.0)]
    line_scale: f32,
    /// Don't draw line geometry (IndexedLineSet, LineSet).
    #[arg(long)]
    hide_lines: bool,
    /// Don't draw PointSets.
    #[arg(long)]
    hide_points: bool,
    /// Tone-mapping curve; anything but `off` renders in HDR first.
    #[arg(long, value_enum, default_value_t = ToneMap::Off)]
    tone_map: ToneMap,
    /// Bloom strength (0 = off; 0.05–0.3 is a good range).
    #[arg(long, default_value_t = 0.0)]
    bloom: f32,
    /// Linear brightness where bloom starts.
    #[arg(long, default_value_t = 0.8)]
    bloom_threshold: f32,
    /// Shadows: lights with X3D shadows='true' (or the first DirectionalLight).
    #[arg(long)]
    shadows: bool,
    /// Frustum culling: off, cpu, or gpu (compute pass + indirect draws).
    #[arg(long, value_enum, default_value_t = Cull::Cpu)]
    cull: Cull,
    /// EnvironmentLight shading: single, split or multi (multiple scattering).
    #[arg(long, value_enum, default_value_t = IblModel::Multi)]
    ibl: IblModel,
    /// Linear → display: srgb (the standard) or gamma22 (what X_ITE writes).
    #[arg(long, value_enum, default_value_t = Encode::Srgb)]
    encode: Encode,
}

impl Common {
    fn settings(&self) -> RenderSettings {
        RenderSettings {
            view: self.view,
            color_mode: self.color_mode,
            wireframe: self.wireframe,
            headlight: self.headlight,
            cull_back_faces: !self.no_cull,
            exposure: self.exposure,
            line_scale: self.line_scale,
            show_lines: !self.hide_lines,
            show_points: !self.hide_points,
            tone_map: self.tone_map,
            bloom: self.bloom,
            bloom_threshold: self.bloom_threshold,
            shadows: self.shadows,
            cull: self.cull,
            ibl: self.ibl,
            encode: self.encode,
        }
    }
    fn load_options(&self) -> LoadOptions {
        LoadOptions { segments: self.segments, strict: self.strict, load_all_inlines: self.load_all_inlines }
    }
}

/// `1280x720` → `Size(1280, 720)`.
/// 🦀 A *newtype* with `FromStr` is all clap needs to parse a custom type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Size(u32, u32);

impl FromStr for Size {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (w, h) = s.split_once(['x', 'X', '×']).ok_or("expected WIDTHxHEIGHT, e.g. 1280x720")?;
        let parse = |v: &str| v.trim().parse::<u32>().map_err(|e| format!("`{v}`: {e}"));
        match (parse(w)?, parse(h)?) {
            (0, _) | (_, 0) => Err("width and height must be > 0".into()),
            (w, h) => Ok(Self(w, h)),
        }
    }
}

fn main() -> Result<()> {
    // RUST_LOG=info (or debug, or wgpu_core=warn,x3d_render=debug) for more detail.
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let cli = Cli::parse();
    match cli.command {
        #[cfg(feature = "window")]
        Command::View { file, shader, screenshot, exit_after, common } => {
            x3d_render::viewer::run(x3d_render::viewer::ViewerOptions {
                scene: file,
                shader,
                backend: common.backend,
                profile: common.profile,
                msaa: common.msaa,
                size: (common.size.0, common.size.1),
                settings: common.settings(),
                load: common.load_options(),
                screenshot,
                exit_after,
            })
        }
        Command::Conform { dir, receipt, backend } => {
            let gpu = Gpu::headless(backend)?;
            let root = dir.parent().map(Path::to_path_buf).unwrap_or_default();
            let r = x3d_render::conform::run(&gpu, &root, &dir)?;
            println!("adapter: {}\nvalidator: {}\n", gpu.describe(), r.json["validator"]);
            print!("{}", r.table());
            println!("\n{}", r.json["summary"]);
            if let Some(path) = receipt {
                std::fs::write(&path, serde_json::to_string_pretty(&r.json)?)?;
                println!("receipt: {}", path.display());
            }
            if !r.passed() {
                bail!("conformance failed");
            }
            Ok(())
        }
        Command::Bench { file, frames, viewpoint, time, common } => {
            let mut scene = x3d_scene::load_with(&file, &common.load_options())
                .with_context(|| format!("loading {}", file.display()))?;
            scene.advance(0.0);
            if let Some(i) = viewpoint {
                scene.bind_viewpoint(i);
                scene.advance(0.0);
            }
            scene.advance(time);
            let gpu = Gpu::headless_with(common.backend, common.profile)?;
            let shot = Shot {
                size: (common.size.0, common.size.1),
                msaa: common.msaa,
                settings: common.settings(),
                viewpoint,
            };
            println!(
                "{} at {}×{}, msaa {}, {:?}",
                file.display(),
                shot.size.0,
                shot.size.1,
                shot.msaa,
                shot.settings.cull
            );
            print!("{}", x3d_render::bench::bench(&gpu, &scene, &shot, frames));
            Ok(())
        }
        Command::Render { file, output, viewpoint, sheet, time, common } => {
            let t0 = Instant::now();
            let mut scene = x3d_scene::load_with(&file, &common.load_options())
                .with_context(|| format!("loading {}", file.display()))?;
            // Like a browser's first frame: animations show their state at `time`.
            scene.advance(0.0); // load: the first Viewpoint is bound, intros start
            if let Some(i) = viewpoint {
                scene.bind_viewpoint(i);
                scene.advance(0.0);
            }
            scene.advance(time);
            let t_load = t0.elapsed();
            if !scene.report.is_lossless() {
                eprint!("{}", scene.report);
            }
            if let Some(i) = viewpoint {
                if i >= scene.viewpoints.len() {
                    bail!("viewpoint {i} doesn't exist; the scene has {}", scene.viewpoints.len());
                }
            }
            let gpu = Gpu::headless_with(common.backend, common.profile)?;
            let shot = Shot {
                size: (common.size.0, common.size.1),
                msaa: common.msaa,
                settings: common.settings(),
                viewpoint,
            };
            let t1 = Instant::now();
            let image = if sheet {
                x3d_render::render_sheet(&gpu, &scene, &shot)?
            } else {
                x3d_render::render_scene(&gpu, &scene, &shot)?
            };
            let t_render = t1.elapsed();
            image.save(&output).with_context(|| format!("writing {}", output.display()))?;
            println!(
                "{} → {} ({}×{}) on {} — load {:.1} ms, render+readback {:.1} ms",
                file.display(),
                output.display(),
                image.width(),
                image.height(),
                gpu.describe(),
                t_load.as_secs_f64() * 1e3,
                t_render.as_secs_f64() * 1e3
            );
            Ok(())
        }
        Command::Info { file, shapes, load_all_inlines } => {
            let t0 = Instant::now();
            let opts = LoadOptions { load_all_inlines, ..Default::default() };
            let scene = x3d_scene::load_with(&file, &opts).with_context(|| format!("loading {}", file.display()))?;
            println!("── {} ── loaded in {:.0} ms\n{}", file.display(), t0.elapsed().as_secs_f64() * 1e3, scene.report);
            let b = scene.bounds();
            if !b.is_empty() {
                println!("bounds {:.3} … {:.3} (size {:.3})", b.min, b.max, b.size());
            }
            for (i, vp) in scene.viewpoints.iter().enumerate() {
                println!("viewpoint {i}: {:?} at {:.2}", vp.description, vp.position);
            }
            if shapes {
                for d in &scene.draws {
                    let m = &d.material;
                    let kind = match (m.unlit, m.pbr.is_some()) {
                        (true, _) => "unlit",
                        (false, true) => "physical",
                        (false, false) => "classic",
                    };
                    println!(
                        "  {:<40} {:>7} {:<9?} {:<8} {} tex  centre {:.1}",
                        d.mesh.label,
                        d.mesh.primitive_count(),
                        d.mesh.topology,
                        kind,
                        m.textures.slots().iter().flatten().count(),
                        d.world_bounds().center()
                    );
                }
            }
            Ok(())
        }
        Command::Shaders { file, show } => {
            let scene = x3d_scene::load(&file).with_context(|| format!("loading {}", file.display()))?;
            let mut failed = 0;
            for (i, shader) in scene.shaders.iter().enumerate() {
                // A shader drawn on a PointSet needs the sprite version.
                let sprites = scene
                    .draws
                    .iter()
                    .any(|d| d.material.shader == Some(i as u32) && d.mesh.topology == x3d_scene::Topology::Points);
                match x3d_render::custom::compile(shader, sprites) {
                    Ok(c) => {
                        let names: Vec<&str> = c.block.members.iter().map(|m| m.name.as_str()).collect();
                        println!(
                            "✔ {}{}: {} uniforms ({})",
                            shader.label,
                            if sprites { " [sprites]" } else { "" },
                            names.len(),
                            names.join(", ")
                        );
                        if show {
                            println!(
                                "--- vertex ---\n{}\n--- fragment ---\n{}",
                                c.translated.vertex, c.translated.fragment
                            );
                        }
                    }
                    Err(e) => {
                        failed += 1;
                        println!("✘ {}{}:\n{e}", shader.label, if sprites { " [sprites]" } else { "" });
                    }
                }
            }
            if failed > 0 {
                bail!("{failed} of {} shaders failed", scene.shaders.len());
            }
            Ok(())
        }
        Command::Adapters { backend, budget, profile } => {
            let adapters = x3d_render::gpu::list_adapters(backend);
            if adapters.is_empty() {
                bail!("no adapters for {backend:?}");
            }
            for a in adapters {
                println!("{}", x3d_render::gpu::describe(&a));
            }
            if budget {
                let gpu = Gpu::headless_with(backend, profile)?;
                println!("\nbudget on {} (profile {profile:?}):\n{}", gpu.describe(), gpu.budget);
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_parse() {
        assert_eq!("800x600".parse::<Size>(), Ok(Size(800, 600)));
        assert_eq!("1920 × 1080".parse::<Size>(), Ok(Size(1920, 1080)));
        assert!("800".parse::<Size>().is_err());
        assert!("0x5".parse::<Size>().is_err());
    }

    #[test]
    fn cli_is_well_formed() {
        // clap checks the whole derive tree for conflicts.
        use clap::CommandFactory;
        Cli::command().debug_assert();
    }
}
