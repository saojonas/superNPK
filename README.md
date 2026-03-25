# Solver de Formulação Organomineral

Aplicação Streamlit para otimizar mistura de fertilizantes organominerais via programação linear.

## O que faz
- Lê catálogo em JSON ou CSV.
- Minimiza custo por tonelada.
- Respeita garantias informadas pelo usuário.
- Força mínimo orgânico por massa (padrão 8%).
- Pode restringir para usar apenas itens em estoque.
- Exporta mistura ótima em CSV e relatório em Excel.

## Como rodar
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Estrutura esperada do CSV
Colunas mínimas úteis:
- produto
- categoria
- origem
- granulometria
- estoque
- valor_comercial_medio_brl_ton
- nutrientes nas colunas:
  - n_sol_agua
  - p2o5_cna_agua
  - p2o5_total
  - k2o_sol_agua
  - carbono_organico
  - ca
  - mg
  - s
  - b
  - zn
  - mn
  - si
  - cu

## Observações importantes
- Itens com preço zero podem ser excluídos por padrão, para não distorcer o ótimo.
- O mínimo orgânico foi modelado por participação mássica de fontes orgânicas (`origem=organica` ou `categoria=organico`).
- O solver atual trabalha com metas mínimas. Ele não pune automaticamente excesso de nutriente.
