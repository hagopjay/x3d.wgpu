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

use std::path::PathBuf;
use std::str::FromStr;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use x3d_render::{Backend, ColorMode, DebugView, Gpu, RenderSettings, Shot};
use x3d_scene::LoadOptions;

#[derive(Parser, Debug)]
#[command(
    name = "x3d-view",
    version,
    about = "A wgpu (Vulkan) renderer for X3D scenes"
)]
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
        /// A 3×2 contact sheet of every debug view instead of one image.
        #[arg(long)]
        sheet: bool,
        #[command(flatten)]
        common: Common,
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
    /// List the GPU adapters wgpu can see.
    Adapters {
        #[arg(long, value_enum, default_value_t = Backend::Auto)]
        backend: Backend,
    },
}

/// Options shared by `view` and `render`.
/// 🦀 `#[command(flatten)]` splices these fields into each subcommand.
#[derive(Args, Debug)]
struct Common {
    #[arg(long, value_enum, default_value_t = Backend::Auto)]
    backend: Backend,
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
    /// Brighten or darken PhysicalMaterial lighting (1 = as X_ITE draws it).
    #[arg(long, default_value_t = 1.0)]
    exposure: f32,
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
            ..Default::default()
        }
    }
    fn load_options(&self) -> LoadOptions {
        LoadOptions {
            segments: self.segments,
            strict: self.strict,
            load_all_inlines: self.load_all_inlines,
        }
    }
}

/// `1280x720` → `Size(1280, 720)`.
/// 🦀 A *newtype* with `FromStr` is all clap needs to parse a custom type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Size(u32, u32);

impl FromStr for Size {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (w, h) = s
            .split_once(['x', 'X', '×'])
            .ok_or("expected WIDTHxHEIGHT, e.g. 1280x720")?;
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
        Command::View {
            file,
            shader,
            screenshot,
            exit_after,
            common,
        } => x3d_render::viewer::run(x3d_render::viewer::ViewerOptions {
            scene: file,
            shader,
            backend: common.backend,
            msaa: common.msaa,
            size: (common.size.0, common.size.1),
            settings: common.settings(),
            load: common.load_options(),
            screenshot,
            exit_after,
        }),
        Command::Render {
            file,
            output,
            viewpoint,
            sheet,
            common,
        } => {
            let t0 = Instant::now();
            let scene = x3d_scene::load_with(&file, &common.load_options())
                .with_context(|| format!("loading {}", file.display()))?;
            let t_load = t0.elapsed();
            if !scene.report.is_lossless() {
                eprint!("{}", scene.report);
            }
            if let Some(i) = viewpoint {
                if i >= scene.viewpoints.len() {
                    bail!(
                        "viewpoint {i} doesn't exist; the scene has {}",
                        scene.viewpoints.len()
                    );
                }
            }
            let gpu = Gpu::headless(common.backend)?;
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
            image
                .save(&output)
                .with_context(|| format!("writing {}", output.display()))?;
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
        Command::Info {
            file,
            shapes,
            load_all_inlines,
        } => {
            let t0 = Instant::now();
            let opts = LoadOptions {
                load_all_inlines,
                ..Default::default()
            };
            let scene = x3d_scene::load_with(&file, &opts)
                .with_context(|| format!("loading {}", file.display()))?;
            println!(
                "── {} ── loaded in {:.0} ms\n{}",
                file.display(),
                t0.elapsed().as_secs_f64() * 1e3,
                scene.report
            );
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
        Command::Adapters { backend } => {
            let adapters = x3d_render::gpu::list_adapters(backend);
            if adapters.is_empty() {
                bail!("no adapters for {backend:?}");
            }
            for a in adapters {
                println!("{}", x3d_render::gpu::describe(&a));
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
