"""Ponto de entrada local do El Niño Monitor."""

import argparse
from contextlib import ExitStack
from pathlib import Path

import xarray as xr

from el_nino_monitor.config import (
    CLIMATOLOGY_END_YEAR,
    CLIMATOLOGY_START_YEAR,
    get_settings,
)
from el_nino_monitor.data.noaa_client import NOAAClient
from el_nino_monitor.processing.sst import (
    calculate_monthly_climatology,
    prepare_published_noaa_climatology,
)
from el_nino_monitor.processing.history import (
    MonthPeriod,
    build_nino34_history,
    calculate_period_statistics,
    compare_periods,
    export_history_csv,
)
from el_nino_monitor.utils.logging import configure_logging
from el_nino_monitor.visualization.maps import save_sst_anomaly_map
from el_nino_monitor.visualization.comparison import save_comparison_map
from el_nino_monitor.visualization.history import (
    save_history_plot,
    save_period_comparison_plot,
)


def _print_statistics(series: xr.Dataset) -> None:
    stats = calculate_period_statistics(series)
    print(f"{stats.area} | {stats.period.label} | referência {stats.reference_period}")
    print(f"Meses válidos: {stats.valid_months}/{stats.total_months}")
    print(f"Anomalia média: {stats.mean_anomaly_c:+.2f} °C")
    print(f"Máxima mensal: {stats.maximum_anomaly_c:+.2f} °C ({stats.maximum_month})")
    print(f"Mínima mensal: {stats.minimum_anomaly_c:+.2f} °C ({stats.minimum_month})")
    if stats.largest_monthly_increase_c is not None:
        print(f"Maior aumento mês a mês: {stats.largest_monthly_increase_c:+.2f} °C")
    if stats.largest_monthly_decrease_c is not None:
        print(f"Maior queda mês a mês: {stats.largest_monthly_decrease_c:+.2f} °C")


def _monthly_paths(
    client: NOAAClient,
    period: MonthPeriod,
    directory: Path,
    *,
    download_missing: bool,
) -> list[Path]:
    paths: list[Path] = []
    for month in period.months:
        filename = f"ersst.v5.{month.year:04d}{month.month:02d}.nc"
        path = directory / filename
        if not path.is_file():
            if not download_missing:
                raise FileNotFoundError(
                    f"Arquivo ausente: {path}; use --download-missing"
                )
            if directory.resolve() != client.settings.raw_dir.resolve():
                raise ValueError(
                    "--download-missing exige --monthly-dir igual a data/raw"
                )
            path = client.download_ncei_month(month.year, month.month)
        paths.append(path)
    return paths


def _run_history_or_comparison(
    args: argparse.Namespace, parser: argparse.ArgumentParser, client: NOAAClient
) -> None:
    if args.start is None or args.end is None:
        parser.error(f"{args.command} requer --start e --end")
    first_period = MonthPeriod(args.start, args.end)
    second_period = None
    if args.command == "compare":
        if args.other_start is None or args.other_end is None:
            parser.error("compare requer --other-start e --other-end")
        second_period = MonthPeriod(args.other_start, args.other_end)
    if args.climatology_path is None and args.path is None:
        parser.error(
            "Arquivos mensais exigem --climatology-path; "
            "a climatologia local requer um arquivo agregado em --path"
        )
    report_dir = args.output_dir
    formats = ("png", "pdf") if args.format == "both" else (args.format,)
    with ExitStack() as stack:
        if args.path is not None:
            first_data = stack.enter_context(client.open_dataset(args.path))
            second_data = first_data
        else:
            directory = args.monthly_dir or client.settings.raw_dir
            first_data = stack.enter_context(
                client.load_monthly_files(
                    _monthly_paths(
                        client, first_period, directory,
                        download_missing=args.download_missing,
                    )
                )
            )
            second_data = (
                stack.enter_context(
                    client.load_monthly_files(
                        _monthly_paths(
                            client, second_period, directory,
                            download_missing=args.download_missing,
                        )
                    )
                )
                if second_period is not None
                else first_data
            )

        if args.climatology_path is not None:
            published = stack.enter_context(
                xr.open_dataset(
                    args.climatology_path, engine="netcdf4",
                    decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
                )
            )
            climatology = prepare_published_noaa_climatology(
                published,
                start_year=args.reference_start,
                end_year=args.reference_end,
            )
        else:
            climatology = calculate_monthly_climatology(
                first_data,
                start_year=args.reference_start,
                end_year=args.reference_end,
            )

        first = build_nino34_history(
            first_data, climatology, first_period, source=args.source
        )
        _print_statistics(first)
        for path in (
            export_history_csv(first, report_dir),
            *save_history_plot(
                first, output_dir=report_dir, formats=formats, dpi=args.dpi
            ),
        ):
            print(path)

        if second_period is None:
            return
        second = build_nino34_history(
            second_data, climatology, second_period, source=args.source
        )
        compare_periods(first, second)
        _print_statistics(second)
        for path in (
            export_history_csv(second, report_dir),
            *save_history_plot(
                second, output_dir=report_dir, formats=formats, dpi=args.dpi
            ),
            *save_period_comparison_plot(
                first, second,
                output_dir=report_dir, formats=formats, dpi=args.dpi,
            ),
        ):
            print(path)
        map_types = ("pacific", "global") if args.map_type == "both" else (args.map_type,)
        for map_type in map_types:
            for path in save_comparison_map(
                first_data, second_data, climatology,
                first_period, second_period,
                map_type=map_type, output_dir=args.maps_dir,
                formats=formats, dpi=args.dpi, source=args.source,
            ):
                print(path)


def main() -> None:
    """Inicializa o projeto ou executa uma operação de coleta/inspeção."""
    parser = argparse.ArgumentParser(description="Monitor local NOAA ERSSTv5")
    parser.add_argument(
        "command", nargs="?",
        choices=("download", "inspect", "map", "history", "compare"),
    )
    parser.add_argument("--force", action="store_true", help="Atualiza o arquivo em cache")
    parser.add_argument("--url", help="Fonte NOAA alternativa escolhida explicitamente")
    parser.add_argument("--path", type=Path, help="NetCDF agregado para inspect, map, history ou compare")
    parser.add_argument("--year", type=int, help="Ano do mapa")
    parser.add_argument("--month", type=int, help="Mês do mapa")
    parser.add_argument(
        "--map-type", choices=("pacific", "global", "both"), default="pacific",
        help="Área do mapa",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Resolução do PNG")
    parser.add_argument("--output-dir", type=Path, help="Saída de mapas (map) ou relatórios (history/compare)")
    parser.add_argument(
        "--format", choices=("png", "pdf", "both"), default="both",
        help="Formato de exportação",
    )
    parser.add_argument(
        "--climatology-path", type=Path,
        help="Climatologia mensal NOAA PSL já publicada; sem esta opção, calcula a partir de --path",
    )
    parser.add_argument("--reference-start", type=int, default=CLIMATOLOGY_START_YEAR)
    parser.add_argument("--reference-end", type=int, default=CLIMATOLOGY_END_YEAR)
    parser.add_argument("--source", default="NOAA ERSSTv5", help="Fonte informada nas saídas")
    parser.add_argument("--start", help="Primeiro mês do período: YYYY-MM")
    parser.add_argument("--end", help="Último mês do período: YYYY-MM")
    parser.add_argument("--other-start", help="Primeiro mês do segundo período")
    parser.add_argument("--other-end", help="Último mês do segundo período")
    parser.add_argument(
        "--monthly-dir", type=Path,
        help="Pasta com arquivos mensais NCEI; padrão: data/raw",
    )
    parser.add_argument(
        "--download-missing", action="store_true",
        help="Baixa explicitamente meses NCEI ausentes",
    )
    parser.add_argument(
        "--maps-dir", type=Path,
        help="Pasta dos mapas comparativos; padrão: outputs/maps",
    )
    args = parser.parse_args()
    settings = get_settings()
    logger = configure_logging(settings.log_level)
    settings.create_directories()
    if args.command is None:
        logger.info("Projeto inicializado; dados em %s", settings.data_dir)
        return

    client = NOAAClient(settings, **({"url": args.url} if args.url else {}))
    if args.command in ("history", "compare"):
        _run_history_or_comparison(args, parser, client)
        return
    if args.command == "map":
        if args.year is None or args.month is None:
            parser.error("map requer --year e --month")
        path = args.path or client.dataset_path
        with client.open_dataset(path) as dataset:
            if args.climatology_path is not None:
                with xr.open_dataset(
                    args.climatology_path,
                    engine="netcdf4",
                    decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
                ) as published:
                    climatology = prepare_published_noaa_climatology(
                        published,
                        start_year=args.reference_start,
                        end_year=args.reference_end,
                    )
            else:
                climatology = calculate_monthly_climatology(
                    dataset,
                    start_year=args.reference_start,
                    end_year=args.reference_end,
                )
            map_types = ("pacific", "global") if args.map_type == "both" else (args.map_type,)
            formats = ("png", "pdf") if args.format == "both" else (args.format,)
            for map_type in map_types:
                paths = save_sst_anomaly_map(
                    dataset, climatology,
                    year=args.year, month=args.month,
                    map_type=map_type, dpi=args.dpi,
                    output_dir=args.output_dir, formats=formats,
                    source=args.source,
                )
                for output in paths:
                    print(output)
        return

    if args.command == "download":
        path = client.download_ersst(force=args.force)
    else:
        path = args.path or client.dataset_path
    with client.open_dataset(path) as dataset:
        metadata = client.get_metadata(dataset)
    print(f"Arquivo: {path}")
    print(f"Dimensões: {metadata.dimensions}")
    print(f"Coordenadas: {metadata.latitude_name}, {metadata.longitude_name}")
    print(f"SST: {metadata.sst_variable} ({metadata.temperature_units})")
    print(f"Período: {metadata.time_start} a {metadata.time_end}")
    print(f"Valores ausentes: {metadata.missing_value_count}")


if __name__ == "__main__":
    main()
