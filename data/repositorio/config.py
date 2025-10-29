from pathlib import Path
root_path = Path(__file__).resolve().parent

figures_path = root_path / 'reports' / 'figures'
tables_path = root_path / 'reports' / 'tables'
visualisation_path = root_path / 'src' / 'visualization'

db_endpoints = {
    'dev': 'prueba1.xlsx',
    'prod': 'main.xlsx',
}