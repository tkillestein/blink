import sys
from pathlib import Path

import rich_click as click
from click.exceptions import Exit
from loguru import logger
from rich.console import Console
from rich.syntax import Syntax

console = Console()

_VERBOSITY = {0: "INFO", 1: "DEBUG"}
_JOB_TYPES = ["pretrain", "etl", "manifest"]


@click.group()
@click.option("-v", "--verbose", count=True, help="Increase verbosity (-v, -vv).")
@click.version_option(package_name="blink")
def cli(verbose: int) -> None:
    """Blink — self-supervised pretraining for astronomical imaging."""
    logger.remove()
    logger.add(
        sink=sys.stderr,
        level=_VERBOSITY.get(verbose, "DEBUG"),
        enqueue=True,
    )
    logger.debug("Logging enabled")


@cli.command("run")
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
def run(config_path: Path) -> None:
    """Run any blink job from a config file."""
    from pydantic import ValidationError

    from blink.config import (
        BlinkConfigFile,
        ETLConfig,
        LeJEPAPretrainConfig,
        ManifestConfig,
    )
    from blink.data import WebDatasetIngestPipeline, gather_manifest
    from blink.pretrain import pretrain

    try:
        cfg = BlinkConfigFile.from_toml(config_path).root
    except ValidationError as e:
        logger.error(f"Invalid config:\n{e}")
        raise Exit(1) from e

    logging_path = cfg.output_dir

    # TODO: avoid DRY here - maybe define `run` paths on cfgs?
    match cfg:
        case LeJEPAPretrainConfig():
            logger.add(sink=logging_path / "pretrain.log", level="DEBUG", enqueue=True)
            pretrain(cfg)
        case ETLConfig():
            logger.add(sink=logging_path / "etl.log", level="DEBUG", enqueue=True)
            WebDatasetIngestPipeline(cfg).ingest()
        case ManifestConfig():
            logger.add(sink=logging_path / "manifest.log", level="DEBUG", enqueue=True)
            gather_manifest(cfg)


@cli.command("validate")
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
def validate(config_path: Path) -> None:
    """Validate any blink config file."""
    from pydantic import ValidationError

    from blink.config import BlinkConfigFile

    try:
        cfg = BlinkConfigFile.from_toml(config_path)
        logger.success(f"Valid config of type: [bold]{type(cfg.root).__name__}[/bold]")
    except ValidationError as e:
        logger.error(f"Invalid config:\n{e}")
        raise Exit(1) from e


@cli.command("dump")
@click.argument("job_type", type=click.Choice(_JOB_TYPES))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Write to file. Prints to stdout if omitted.",
)
def dump(job_type: str, output: Path | None) -> None:
    """Dump a blank config template for a given job type."""
    from blink.config import ETLConfig, LeJEPAPretrainConfig, ManifestConfig

    content = None

    match job_type:
        case "pretrain":
            content = LeJEPAPretrainConfig.empty().to_toml()
        case "etl":
            content = ETLConfig.empty().to_toml()
        case "manifest":
            content = ManifestConfig.empty().to_toml()
        case _:
            msg = f"Unknown job type: {job_type}"
            raise RuntimeError(msg)

    if output is None:
        console.print(Syntax(content, "toml"))
    else:
        output.write_text(content)
        logger.success(f"Config written to {output}")


@cli.command("init")
@click.argument("job_type", type=click.Choice(_JOB_TYPES))
def init(job_type: str) -> None:
    """Initialize a blink job, populating folder structure."""
    from blink.config import ETLConfig, LeJEPAPretrainConfig, ManifestConfig

    config = None

    match job_type:
        case "pretrain":
            config = LeJEPAPretrainConfig.empty()
        case "etl":
            config = ETLConfig.empty()

            if matches := sorted(config.output_dir.glob("*.zst")):
                logger.info(f"Automatically identified {matches[0]} as manifest file")
                config.manifest_path = matches[0]

        case "manifest":
            config = ManifestConfig.empty()
        case _:
            msg = f"Unknown job type: {job_type}"
            raise RuntimeError(msg)

    if not config.output_dir.exists():
        config.output_dir.mkdir(parents=True)

    output_path = config.output_dir / f"{job_type}_config.toml"
    config.write_toml(output_path=output_path)
    logger.success(f"Config file written to {output_path}")
