from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import linprog

APP_DIR = Path(__file__).resolve().parent
DEFAULT_JSON = APP_DIR / 'catalogo_materia_prima.json'

NUTRIENT_KEYS = [
    'n_sol_agua',
    'p2o5_cna_agua',
    'p2o5_total',
    'k2o_sol_agua',
    'carbono_organico',
    'ca',
    'mg',
    's',
    'b',
    'zn',
    'mn',
    'si',
    'cu',
]

NUTRIENT_LABELS = {
    'n_sol_agua': 'N sol. água (%)',
    'p2o5_cna_agua': 'P2O5 CNA+água (%)',
    'p2o5_total': 'P2O5 total (%)',
    'k2o_sol_agua': 'K2O sol. água (%)',
    'carbono_organico': 'Carbono orgânico (%)',
    'ca': 'Ca (%)',
    'mg': 'Mg (%)',
    's': 'S (%)',
    'b': 'B (%)',
    'zn': 'Zn (%)',
    'mn': 'Mn (%)',
    'si': 'Si (%)',
    'cu': 'Cu (%)',
}

DISPLAY_COLUMNS = [
    'produto', 'categoria', 'origem', 'granulometria', 'estoque', 'valor_comercial_medio_brl_ton',
    *NUTRIENT_KEYS,
]


@dataclass
class SolveResult:
    success: bool
    message: str
    mix_df: pd.DataFrame | None = None
    guarantee_df: pd.DataFrame | None = None
    unit_cost_brl_ton: float | None = None
    total_cost_brl: float | None = None
    solver_status: str | None = None


def flatten_product(prod: Dict) -> Dict:
    macros = prod.get('nutrientes', {}).get('macros_secundarios', {}) or {}
    micros = prod.get('nutrientes', {}).get('micros', {}) or {}
    nut = prod.get('nutrientes', {}) or {}

    row = {
        'produto': prod.get('produto', ''),
        'categoria': prod.get('categoria', ''),
        'granulometria': prod.get('granulometria', ''),
        'origem': prod.get('origem', ''),
        'analise_variavel': bool(prod.get('analise_variavel', False)),
        'estoque': bool(prod.get('estoque', False)),
        'valor_comercial_medio_brl_ton': float(prod.get('valor_comercial_medio_brl_ton', 0) or 0),
        'n_sol_agua': float(nut.get('n_sol_agua', 0) or 0),
        'p2o5_cna_agua': float(nut.get('p2o5_cna_agua', 0) or 0),
        'p2o5_total': float(nut.get('p2o5_total', 0) or 0),
        'k2o_sol_agua': float(nut.get('k2o_sol_agua', 0) or 0),
        'carbono_organico': float(nut.get('carbono_organico', 0) or 0),
        'ca': float(macros.get('ca', 0) or 0),
        'mg': float(macros.get('mg', 0) or 0),
        's': float(macros.get('s', 0) or 0),
        'b': float(micros.get('b', 0) or 0),
        'zn': float(micros.get('zn', 0) or 0),
        'mn': float(micros.get('mn', 0) or 0),
        'si': float(micros.get('si', 0) or 0),
        'cu': float(micros.get('cu', 0) or 0),
    }
    return row


def load_json_catalog(file) -> pd.DataFrame:
    data = json.load(file)
    produtos = data.get('produtos', [])
    if not produtos:
        raise ValueError('JSON sem a chave "produtos" ou sem itens.')
    rows = [flatten_product(p) for p in produtos]
    df = pd.DataFrame(rows)
    return normalize_dataframe(df)


def load_csv_catalog(file) -> pd.DataFrame:
    # Tenta UTF-8 e fallback latin1
    raw = file.read()
    if isinstance(raw, bytes):
        for enc in ('utf-8', 'latin1', 'cp1252'):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError('Não foi possível decodificar o CSV.')
        from io import StringIO
        df = pd.read_csv(StringIO(text))
    else:
        df = pd.read_csv(file)
    return normalize_dataframe(df)


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    required_base = ['produto', 'categoria', 'origem', 'granulometria', 'estoque', 'valor_comercial_medio_brl_ton']
    for col in required_base:
        if col not in df.columns:
            df[col] = '' if col in ['produto', 'categoria', 'origem', 'granulometria'] else 0

    for col in NUTRIENT_KEYS:
        if col not in df.columns:
            df[col] = 0.0

    df['estoque'] = df['estoque'].apply(_to_bool)
    num_cols = ['valor_comercial_medio_brl_ton', *NUTRIENT_KEYS]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    if 'produto' not in df.columns or df['produto'].astype(str).str.strip().eq('').all():
        raise ValueError('O catálogo precisa ter a coluna/campo "produto".')

    return df


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    return s in {'1', 'true', 'sim', 'yes', 'y'}




def parse_formula_text(formula: str) -> Dict[str, float]:
    s = (formula or '').strip().lower().replace(' ', '')
    if not s:
        return {}

    targets = {k: 0.0 for k in NUTRIENT_KEYS}

    # aceita formatos como 10-10-10, 10-10-10+2%b, 04-14-08+0.3%zn
    if '-' in s:
        base, *extras = s.split('+')
        parts = base.split('-')
        if len(parts) >= 1 and parts[0]:
            targets['n_sol_agua'] = float(parts[0].replace('%', '').replace(',', '.'))
        if len(parts) >= 2 and parts[1]:
            targets['p2o5_cna_agua'] = float(parts[1].replace('%', '').replace(',', '.'))
            targets['p2o5_total'] = max(targets['p2o5_total'], targets['p2o5_cna_agua'])
        if len(parts) >= 3 and parts[2]:
            targets['k2o_sol_agua'] = float(parts[2].replace('%', '').replace(',', '.'))

        alias = {
            'b': 'b', 'zn': 'zn', 'mn': 'mn', 'si': 'si', 'cu': 'cu',
            'ca': 'ca', 'mg': 'mg', 's': 's', 'corg': 'carbono_organico',
            'carbonoorganico': 'carbono_organico', 'carbono_org': 'carbono_organico'
        }
        for extra in extras:
            token = extra.replace('%', '')
            i = 0
            while i < len(token) and (token[i].isdigit() or token[i] in ',.'):
                i += 1
            if i == 0:
                continue
            value = float(token[:i].replace(',', '.'))
            key = alias.get(token[i:])
            if key:
                targets[key] = value
        return targets

    raise ValueError('Formato de formulação inválido. Use, por exemplo: 10-10-10+2%b')

def build_template_csv() -> bytes:
    df = pd.DataFrame([
        {
            'produto': 'Ureia46',
            'categoria': 'nitrogenado',
            'origem': 'mineral',
            'granulometria': 'granulada',
            'estoque': True,
            'valor_comercial_medio_brl_ton': 3670,
            'n_sol_agua': 46,
            'p2o5_cna_agua': 0,
            'p2o5_total': 0,
            'k2o_sol_agua': 0,
            'carbono_organico': 0,
            'ca': 0,
            'mg': 0,
            's': 0,
            'b': 0,
            'zn': 0,
            'mn': 0,
            'si': 0,
            'cu': 0,
        },
        {
            'produto': 'CompostoOrganicoBase',
            'categoria': 'organico',
            'origem': 'organica',
            'granulometria': 'farelada',
            'estoque': False,
            'valor_comercial_medio_brl_ton': 220,
            'n_sol_agua': 0.8,
            'p2o5_cna_agua': 0.6,
            'p2o5_total': 1,
            'k2o_sol_agua': 0.4,
            'carbono_organico': 20,
            'ca': 5,
            'mg': 0.2,
            's': 0.4,
            'b': 0,
            'zn': 0.02,
            'mn': 0,
            'si': 11,
            'cu': 0,
        },
    ])
    return df.to_csv(index=False).encode('utf-8')




def export_catalog_csv(df: pd.DataFrame) -> bytes:
    return df[DISPLAY_COLUMNS].to_csv(index=False).encode('utf-8')


def render_catalog_editor(df: pd.DataFrame) -> pd.DataFrame:
    editor_df = st.data_editor(
        df[DISPLAY_COLUMNS].copy(),
        use_container_width=True,
        height=420,
        hide_index=True,
        num_rows='fixed',
        column_config={
            'produto': st.column_config.TextColumn('Produto', disabled=True),
            'categoria': st.column_config.TextColumn('Categoria'),
            'origem': st.column_config.TextColumn('Origem'),
            'granulometria': st.column_config.TextColumn('Granulometria'),
            'estoque': st.column_config.CheckboxColumn('Em estoque?'),
            'valor_comercial_medio_brl_ton': st.column_config.NumberColumn('Valor (R$/t)', min_value=0.0, step=1.0, format='%.2f'),
            'n_sol_agua': st.column_config.NumberColumn('N (%)', format='%.4f'),
            'p2o5_cna_agua': st.column_config.NumberColumn('P2O5 CNA+água (%)', format='%.4f'),
            'p2o5_total': st.column_config.NumberColumn('P2O5 total (%)', format='%.4f'),
            'k2o_sol_agua': st.column_config.NumberColumn('K2O (%)', format='%.4f'),
            'carbono_organico': st.column_config.NumberColumn('C orgânico (%)', format='%.4f'),
            'ca': st.column_config.NumberColumn('Ca (%)', format='%.4f'),
            'mg': st.column_config.NumberColumn('Mg (%)', format='%.4f'),
            's': st.column_config.NumberColumn('S (%)', format='%.4f'),
            'b': st.column_config.NumberColumn('B (%)', format='%.4f'),
            'zn': st.column_config.NumberColumn('Zn (%)', format='%.4f'),
            'mn': st.column_config.NumberColumn('Mn (%)', format='%.4f'),
            'si': st.column_config.NumberColumn('Si (%)', format='%.4f'),
            'cu': st.column_config.NumberColumn('Cu (%)', format='%.4f'),
        },
        key='catalogo_editor',
    )

    editor_df = normalize_dataframe(editor_df)
    return editor_df


def build_bounds(df: pd.DataFrame, use_inventory_only: bool, allow_zero_price: bool) -> Tuple[pd.DataFrame, List[Tuple[float, float | None]], List[str]]:
    work = df.copy()
    removed = []

    if use_inventory_only:
        mask = work['estoque'] == True
        removed.extend(work.loc[~mask, 'produto'].tolist())
        work = work.loc[mask].copy()

    if not allow_zero_price:
        zero_mask = work['valor_comercial_medio_brl_ton'] <= 0
        removed.extend(work.loc[zero_mask, 'produto'].tolist())
        work = work.loc[~zero_mask].copy()

    bounds = [(0, None) for _ in range(len(work))]
    return work.reset_index(drop=True), bounds, removed


def solve_formula(
    df: pd.DataFrame,
    targets: Dict[str, float],
    batch_mass_kg: float,
    min_organic_pct: float,
    use_inventory_only: bool,
    allow_zero_price: bool,
    prefer_inventory_bonus_brl_ton: float,
) -> SolveResult:
    if batch_mass_kg <= 0:
        return SolveResult(False, 'A massa do lote precisa ser maior que zero.')

    work, bounds, removed = build_bounds(df, use_inventory_only, allow_zero_price)
    if work.empty:
        return SolveResult(False, 'Nenhuma matéria-prima elegível sobrou após os filtros aplicados.')

    n = len(work)
    c = work['valor_comercial_medio_brl_ton'].to_numpy(dtype=float) / 1000.0
    stock_discount = np.where(work['estoque'].to_numpy(), prefer_inventory_bonus_brl_ton / 1000.0, 0.0)
    c = c - stock_discount

    A_eq = [np.ones(n)]
    b_eq = [batch_mass_kg]

    A_ub = []
    b_ub = []

    # Garantias mínimas
    for key, target_pct in targets.items():
        if target_pct > 0:
            coeff = -work[key].to_numpy(dtype=float) / 100.0
            A_ub.append(coeff)
            b_ub.append(-(target_pct / 100.0) * batch_mass_kg)

    # mínimo de composto orgânico por massa de fontes orgânicas
    organic_mask = work['origem'].astype(str).str.lower().eq('organica') | work['categoria'].astype(str).str.lower().eq('organico')
    if organic_mask.any() and min_organic_pct > 0:
        coeff = np.where(organic_mask.to_numpy(), -1.0, 0.0)
        A_ub.append(coeff)
        b_ub.append(-(min_organic_pct / 100.0) * batch_mass_kg)
    elif min_organic_pct > 0:
        return SolveResult(False, 'Não existe nenhuma matéria-prima orgânica elegível para cumprir o mínimo orgânico.')

    res = linprog(
        c=c,
        A_ub=np.array(A_ub) if A_ub else None,
        b_ub=np.array(b_ub) if b_ub else None,
        A_eq=np.array(A_eq),
        b_eq=np.array(b_eq),
        bounds=bounds,
        method='highs',
    )

    if not res.success:
        msg = res.message
        if removed:
            msg += f' | Removidos pelos filtros/preço: {", ".join(sorted(set(removed)))}'
        return SolveResult(False, msg, solver_status=str(res.status))

    x = res.x
    mix_df = work.copy()
    mix_df['kg'] = x
    mix_df['participacao_pct'] = (mix_df['kg'] / batch_mass_kg) * 100.0
    mix_df['custo_brl'] = mix_df['kg'] * (mix_df['valor_comercial_medio_brl_ton'] / 1000.0)
    mix_df = mix_df.loc[mix_df['kg'] > 1e-6].copy()
    mix_df = mix_df.sort_values(['kg', 'valor_comercial_medio_brl_ton'], ascending=[False, True]).reset_index(drop=True)

    guarantee_rows = []
    for key in NUTRIENT_KEYS:
        achieved_kg = float((work[key].to_numpy(dtype=float) / 100.0 * x).sum())
        achieved_pct = (achieved_kg / batch_mass_kg) * 100.0
        guarantee_rows.append({
            'garantia': NUTRIENT_LABELS[key],
            'meta_%': float(targets.get(key, 0.0)),
            'atingido_%': achieved_pct,
            'folga_%': achieved_pct - float(targets.get(key, 0.0)),
        })

    organic_kg = float(x[organic_mask.to_numpy()].sum())
    organic_pct = (organic_kg / batch_mass_kg) * 100.0
    guarantee_rows.append({
        'garantia': 'Mínimo orgânico por massa (%)',
        'meta_%': min_organic_pct,
        'atingido_%': organic_pct,
        'folga_%': organic_pct - min_organic_pct,
    })

    guarantee_df = pd.DataFrame(guarantee_rows)
    total_cost = float((work['valor_comercial_medio_brl_ton'].to_numpy(dtype=float) / 1000.0 * x).sum())
    unit_cost = (total_cost / batch_mass_kg) * 1000.0

    return SolveResult(
        True,
        'Solução ótima encontrada.',
        mix_df=mix_df,
        guarantee_df=guarantee_df,
        unit_cost_brl_ton=unit_cost,
        total_cost_brl=total_cost,
        solver_status=str(res.status),
    )


def export_mix_csv(result: SolveResult) -> bytes:
    assert result.mix_df is not None
    cols = ['produto', 'categoria', 'origem', 'granulometria', 'estoque', 'kg', 'participacao_pct', 'valor_comercial_medio_brl_ton', 'custo_brl']
    return result.mix_df[cols].to_csv(index=False).encode('utf-8')


def export_report_excel(result: SolveResult) -> bytes:
    from io import BytesIO
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        assert result.mix_df is not None and result.guarantee_df is not None
        result.mix_df.to_excel(writer, index=False, sheet_name='mistura_otima')
        result.guarantee_df.to_excel(writer, index=False, sheet_name='garantias')
    return output.getvalue()


def main():
    st.set_page_config(page_title='Solver de Formulação Organomineral', layout='wide')
    st.title('SuperNPK - Solver de Formulação Organomineral')
    st.caption('Minimiza custo respeitando garantias nutricionais e mínimo de 8% orgânico por massa.')

    with st.sidebar:
        st.header('Catálogo de matérias-primas')
        source = st.radio('Fonte do catálogo', ['JSON padrão', 'Upload JSON', 'Upload CSV'])

        try:
            if source == 'JSON padrão':
                with open(DEFAULT_JSON, 'r', encoding='utf-8') as f:
                    df = load_json_catalog(f)
            elif source == 'Upload JSON':
                uploaded = st.file_uploader('Envie um JSON', type=['json'])
                if uploaded is None:
                    st.stop()
                df = load_json_catalog(uploaded)
            else:
                uploaded = st.file_uploader('Envie um CSV', type=['csv'])
                st.download_button('Baixar template CSV', build_template_csv(), file_name='template_materias_primas.csv', mime='text/csv')
                if uploaded is None:
                    st.stop()
                df = load_csv_catalog(uploaded)
        except Exception as e:
            st.error(f'Falha ao carregar catálogo: {e}')
            st.stop()

        st.success(f'{len(df)} matérias-primas carregadas.')
        use_inventory_only = st.checkbox('Usar somente itens em estoque', value=True)
        allow_zero_price = st.checkbox('Permitir itens com preço zero', value=False)
        prefer_inventory_bonus_brl_ton = st.number_input('Bônus para priorizar estoque (R$/t)', min_value=0.0, value=15.0, step=5.0)
        batch_mass_kg = st.number_input('Massa do lote (kg)', min_value=1.0, value=1000.0, step=100.0)
        min_organic_pct = st.number_input('Mínimo orgânico por massa (%)', min_value=0.0, value=8.0, step=1.0)

    if 'catalogo_df' not in st.session_state:
        st.session_state['catalogo_df'] = df[DISPLAY_COLUMNS].copy()

    current_catalog_signature = df[DISPLAY_COLUMNS].to_json(orient='split', date_format='iso')
    previous_catalog_signature = st.session_state.get('catalogo_source_signature')
    if previous_catalog_signature != current_catalog_signature:
        st.session_state['catalogo_df'] = df[DISPLAY_COLUMNS].copy()
        st.session_state['catalogo_source_signature'] = current_catalog_signature

    st.subheader('Catálogo normalizado')
    quick1, quick2, quick3 = st.columns([1, 1, 2])
    with quick1:
        if st.button('Marcar todos em estoque'):
            st.session_state['catalogo_df']['estoque'] = True
    with quick2:
        if st.button('Desmarcar todos do estoque'):
            st.session_state['catalogo_df']['estoque'] = False
    with quick3:
        st.download_button(
            'Baixar catálogo editado (CSV)',
            export_catalog_csv(st.session_state['catalogo_df']),
            file_name='catalogo_editado.csv',
            mime='text/csv',
        )

    df = render_catalog_editor(st.session_state['catalogo_df'])
    st.session_state['catalogo_df'] = df.copy()

    st.subheader('Garantias alvo')
    formula_txt = st.text_input('Formulação alvo (atalho)', value='10-10-10+2%b', help='Ex.: 10-10-10, 04-14-08, 10-10-10+2%b, 08-20-10+0.3%zn')
    parsed = {}
    if formula_txt.strip():
        try:
            parsed = parse_formula_text(formula_txt)
            st.caption('Formulação interpretada automaticamente. Você ainda pode ajustar os campos abaixo manualmente.')
        except Exception as e:
            st.warning(f'Não consegui interpretar a formulação digitada: {e}')

    st.write('Informe apenas o que precisa garantir. O restante pode ficar zerado.')

    cols = st.columns(4)
    targets = {}
    for i, key in enumerate(NUTRIENT_KEYS):
        with cols[i % 4]:
            targets[key] = st.number_input(NUTRIENT_LABELS[key], min_value=0.0, value=float(parsed.get(key, 0.0)), step=0.1, format='%.2f')

    if st.button('Resolver formulação', type='primary'):
        result = solve_formula(
            df=df,
            targets=targets,
            batch_mass_kg=batch_mass_kg,
            min_organic_pct=min_organic_pct,
            use_inventory_only=use_inventory_only,
            allow_zero_price=allow_zero_price,
            prefer_inventory_bonus_brl_ton=prefer_inventory_bonus_brl_ton,
        )

        if not result.success:
            st.error(result.message)
            return

        assert result.mix_df is not None and result.guarantee_df is not None
        k1, k2, k3 = st.columns(3)
        k1.metric('Custo estimado (R$/t)', f'{result.unit_cost_brl_ton:,.2f}')
        k2.metric('Custo total do lote (R$)', f'{result.total_cost_brl:,.2f}')
        k3.metric('Itens usados', int(len(result.mix_df)))

        st.subheader('Mistura ótima')
        show_cols = ['produto', 'categoria', 'origem', 'granulometria', 'estoque', 'kg', 'participacao_pct', 'valor_comercial_medio_brl_ton', 'custo_brl']
        st.dataframe(result.mix_df[show_cols], use_container_width=True)

        st.subheader('Checagem das garantias')
        st.dataframe(result.guarantee_df, use_container_width=True)

        st.download_button('Baixar mistura ótima (CSV)', export_mix_csv(result), file_name='mistura_otima.csv', mime='text/csv')
        st.download_button('Baixar relatório (Excel)', export_report_excel(result), file_name='relatorio_formula.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

        st.info(
            'Modelo atual: minimização linear de custo. Quando existir mais de uma solução equivalente, '\
            'o solver tende a escolher uma combinação extrema viável. '\
            'Se você quiser, o próximo passo é eu te entregar a versão com penalização por excesso de garantia, '\
            'limites por matéria-prima, preferência por granulometria e travas operacionais de processo.'
        )


if __name__ == '__main__':
    main()
