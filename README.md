# El Niño Monitor

Aplicação local em Python para analisar a temperatura da superfície do mar (SST) no Pacífico equatorial, com foco na região Niño 3.4. Usa dados mensais NOAA ERSSTv5 para calcular anomalias, gerar séries e mapas e comparar períodos. Os resultados podem ser consultados em um dashboard ou pela API.

## Para que serve

- Acompanhar a SST média e a anomalia mensal na região Niño 3.4.
- Visualizar mapas de anomalias no Pacífico ou no oceano global.
- Exportar séries históricas em CSV e gráficos em PNG/PDF.
- Comparar dois períodos de mesma duração com a mesma referência climatológica.

O processamento é local. A API e o dashboard leem os arquivos disponíveis em `data/raw/` e não atualizam os dados automaticamente.

## Instalação

Requer **Windows, Python 3.12 e pip**. No PowerShell, a partir da pasta do projeto:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

## Como usar

### Preparar os dados e gerar uma série

Este exemplo usa arquivos mensais do [NOAA NCEI](https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/netcdf/) e a [climatologia ERSSTv5 1991–2020 da NOAA PSL](https://downloads.psl.noaa.gov/Datasets/noaa.ersst.v5/sst.mon.ltm.1991-2020.nc):

```powershell
New-Item -ItemType Directory -Force data/raw | Out-Null
Invoke-WebRequest "https://downloads.psl.noaa.gov/Datasets/noaa.ersst.v5/sst.mon.ltm.1991-2020.nc" -OutFile "data/raw/sst.mon.ltm.1991-2020.nc"
.\.venv\Scripts\python.exe main.py history --start 1997-11 --end 1998-01 --download-missing --climatology-path data/raw/sst.mon.ltm.1991-2020.nc
```

`history` baixa os meses ausentes, calcula a série Niño 3.4 e salva CSV e gráficos em `outputs/reports/`. Para comparar duas janelas:

```powershell
.\.venv\Scripts\python.exe main.py compare --start 1997-11 --end 1998-01 --other-start 2015-11 --other-end 2016-01 --download-missing --climatology-path data/raw/sst.mon.ltm.1991-2020.nc
```

`compare` também gera mapas em `outputs/maps/`. Consulte as demais opções com `.\.venv\Scripts\python.exe main.py --help`.

### Abrir o dashboard

Com os dados locais preparados, execute em dois terminais:

```powershell
# Terminal 1: API
.\.venv\Scripts\python.exe -m uvicorn el_nino_monitor.api.main:app --reload
```

```powershell
# Terminal 2: dashboard
.\.venv\Scripts\python.exe -m streamlit run dashboard.py
```

Acesse [http://localhost:8501](http://localhost:8501). A documentação da API fica em [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). O painel mostra indicadores, mapas, série histórica e comparação conforme os meses disponíveis localmente.

### Arquivo agregado opcional

`.\.venv\Scripts\python.exe main.py download` obtém o arquivo mensal agregado ERSSTv5 da NOAA PSL em `data/raw/sst.mnmean.nc`. Use `main.py inspect` para conferir sua cobertura. Com esse arquivo, a aplicação pode calcular a climatologia diretamente dos dados, sem o arquivo de referência publicado.

## Conclusões e limites

A anomalia é a SST observada menos a média do mesmo mês na climatologia de referência. A média regional Niño 3.4 usa pesos proporcionais a `cos(latitude)`; a média móvel de três meses é centrada.

Na comparação documentada com ERSSTv5 e referência 1991–2020, as anomalias médias Niño 3.4 foram **+2,35 °C** entre novembro de 1997 e janeiro de 1998 e **+2,64 °C** entre novembro de 2015 e janeiro de 2016. Esses valores descrevem apenas as janelas escolhidas. A aplicação não classifica episódios oficialmente nem atribui impactos climáticos ao El Niño. Sua média móvel não é o índice ONI/RONI oficial.

Fontes: [NOAA NCEI ERSSTv5](https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/netcdf/) e [NOAA PSL](https://www.psl.noaa.gov/data/gridded/data.noaa.ersst.v5.html).