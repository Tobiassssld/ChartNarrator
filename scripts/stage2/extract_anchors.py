#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract Stage 2 morphology-aware anchors from Stage 1 anchors and cleaned CSV data.

This script builds the enhanced Stage 2 anchor dictionary used by the Stage 2
narrative-generation pipeline. It preserves the original morphology-aware and
binary phase-count logic while aligning paths and messages with the public
repository structure.

Inputs:
    data/stage1/chart_anchors_stage1.json
    data/stage2/candidate_pool/semantic_pool.json
    data/cleaned_csv_data/*.csv

Outputs:
    data/stage2/anchors/chart_anchors_stage2_v2_morph_binary.json

Run from the repository root:
    python scripts/stage2/extract_anchors.py
    python scripts/stage2/extract_anchors.py --dry_run --limit 20
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE1_ANCHOR_FILE = PROJECT_ROOT / "data" / "stage1" / "chart_anchors_stage1.json"
DEFAULT_SEMANTIC_POOL_FILE = PROJECT_ROOT / "data" / "stage2" / "candidate_pool" / "semantic_pool.json"
DEFAULT_CSV_DIR = PROJECT_ROOT / "data" / "cleaned_csv_data"
DEFAULT_STAGE2_ANCHOR_FILE = PROJECT_ROOT / "data" / "stage2" / "anchors" / "chart_anchors_stage2_v2_morph_binary.json"

# ----------------------------------------------------------------
# Variable semantics mapping
# Defines unit description, direction semantics, category, and value nature.
# ----------------------------------------------------------------
# Variable semantics mapping
# Defines unit description, direction semantics, category, and value nature.
# ----------------------------------------------------------------

VARIABLE_SEMANTICS: Dict[str, dict] = {'cbs_Bankruptcies_Key_Figures_Companies.csv': {'topic_category': 'bankruptcies',
                                                'unit_description': 'number of company bankruptcies '
                                                                    '(12-month rolling sum)',
                                                'value_nature': 'flow_count',
                                                'higher_means': 'worse',
                                                'direction_note': 'Rising values indicate increasing '
                                                                  'business failures; declining values '
                                                                  'indicate improving financial conditions.',
                                                'closed_system_note': 'Interpret as structural dynamics '
                                                                      'within the business population only. '
                                                                      'Do not reference specific economic '
                                                                      'crises.'},
 'cbs_Bankruptcies_Key_Figures_Individuals.csv': {'topic_category': 'bankruptcies',
                                                  'unit_description': 'number of individual bankruptcies '
                                                                      '(12-month rolling sum)',
                                                  'value_nature': 'flow_count',
                                                  'higher_means': 'worse',
                                                  'direction_note': 'Rising values indicate increasing '
                                                                    'personal insolvencies.',
                                                  'closed_system_note': 'Interpret as structural dynamics '
                                                                        'within the individual debtor '
                                                                        'population.'},
 'cbs_Bankruptcies_Key_Figures_Total.csv': {'topic_category': 'bankruptcies',
                                            'unit_description': 'total bankruptcies companies + individuals '
                                                                '(12-month rolling sum)',
                                            'value_nature': 'flow_count',
                                            'higher_means': 'worse',
                                            'direction_note': 'Rising values indicate worsening insolvency '
                                                              'conditions across the economy.',
                                            'closed_system_note': 'Interpret as aggregate structural '
                                                                  'dynamics only.'},
 'cbs_Labour_Participation_Unemployment_Total_15_74.csv': {'topic_category': 'labour_market',
                                                           'unit_description': 'unemployment rate, '
                                                                               'percentage of labour force '
                                                                               'aged 15–74',
                                                           'value_nature': 'rate_percentage',
                                                           'higher_means': 'worse',
                                                           'direction_note': 'This is an UNEMPLOYMENT RATE. '
                                                                             'Declining values mean fewer '
                                                                             'people are unemployed — an '
                                                                             'improvement in labour market '
                                                                             'conditions. Do NOT interpret a '
                                                                             'decline as reduced '
                                                                             'participation.',
                                                           'closed_system_note': 'Interpret as structural '
                                                                                 'labour market dynamics '
                                                                                 'only.'},
 'cbs_Labour_Participation_Unemployment_Men_15_74.csv': {'topic_category': 'labour_market',
                                                         'unit_description': 'male unemployment rate, '
                                                                             'percentage of male labour '
                                                                             'force aged 15–74',
                                                         'value_nature': 'rate_percentage',
                                                         'higher_means': 'worse',
                                                         'direction_note': 'Declining values = lower male '
                                                                           'unemployment = improved male '
                                                                           'labour market conditions.',
                                                         'closed_system_note': 'Interpret as structural '
                                                                               'labour market dynamics '
                                                                               'only.'},
 'cbs_Labour_Participation_Unemployment_Women_15_74.csv': {'topic_category': 'labour_market',
                                                           'unit_description': 'female unemployment rate, '
                                                                               'percentage of female labour '
                                                                               'force aged 15–74',
                                                           'value_nature': 'rate_percentage',
                                                           'higher_means': 'worse',
                                                           'direction_note': 'Declining values = lower '
                                                                             'female unemployment = improved '
                                                                             'female labour market '
                                                                             'conditions.',
                                                           'closed_system_note': 'Interpret as structural '
                                                                                 'labour market dynamics '
                                                                                 'only.'},
 'cbs_Inflation_CPI_All_Items.csv': {'topic_category': 'inflation',
                                     'unit_description': 'consumer price index, all items (base year = 100)',
                                     'value_nature': 'index',
                                     'higher_means': 'neutral',
                                     'direction_note': 'This is a PRICE INDEX. Rising values indicate rising '
                                                       'general price levels (inflation). Values above 100 '
                                                       'indicate prices above the base year.',
                                     'closed_system_note': 'Do not attribute price movements to specific '
                                                           'external events or policies.'},
 'cbs_Inflation_CPI_Health.csv': {'topic_category': 'inflation',
                                  'unit_description': 'consumer price index, health category (base year = '
                                                      '100)',
                                  'value_nature': 'index',
                                  'higher_means': 'neutral',
                                  'direction_note': 'This is a PRICE INDEX for the health expenditure '
                                                    'category only. Do not infer overall inflation trends.',
                                  'closed_system_note': 'Do not reference specific healthcare reforms or '
                                                        'policy changes.'},
 'cbs_Inflation_CPI_Housing.csv': {'topic_category': 'inflation',
                                   'unit_description': 'consumer price index, housing category (base year = '
                                                       '100)',
                                   'value_nature': 'index',
                                   'higher_means': 'neutral',
                                   'direction_note': 'This is a PRICE INDEX for housing costs only.',
                                   'closed_system_note': 'Do not reference specific housing market events.'},
 'cbs_PPI_Output_Total.csv': {'topic_category': 'inflation',
                              'unit_description': 'producer price index, total output (base year = 100)',
                              'value_nature': 'index',
                              'higher_means': 'neutral',
                              'direction_note': 'This is a PRODUCER PRICE INDEX. Rising values indicate '
                                                'rising output prices at the producer level.',
                              'closed_system_note': 'Do not reference specific commodity price events.'},
 'cbs_Wages_CAO_Total_Private_Sector.csv': {'topic_category': 'wages',
                                            'unit_description': 'collective labour agreement wage index, '
                                                                'private sector (base year = 100)',
                                            'value_nature': 'index',
                                            'higher_means': 'better',
                                            'direction_note': 'This is a WAGE INDEX. Rising values indicate '
                                                              'higher negotiated wages relative to the base '
                                                              'year.',
                                            'closed_system_note': 'Do not reference specific wage '
                                                                  'negotiations or union agreements.'},
 'cbs_Wages_CAO_Total_Government.csv': {'topic_category': 'wages',
                                        'unit_description': 'collective labour agreement wage index, '
                                                            'government sector (base year = 100)',
                                        'value_nature': 'index',
                                        'higher_means': 'better',
                                        'direction_note': 'Rising values indicate higher negotiated wages in '
                                                          'the government sector.',
                                        'closed_system_note': 'Do not reference specific government wage '
                                                              'policies.'},
 'cbs_Wages_CAO_Total_Total_Monthly_Incl.csv': {'topic_category': 'wages',
                                                'unit_description': 'collective labour agreement wage index, '
                                                                    'total (including all sectors, base year '
                                                                    '= 100)',
                                                'value_nature': 'index',
                                                'higher_means': 'better',
                                                'direction_note': 'Rising values indicate higher negotiated '
                                                                  'wages across all sectors.',
                                                'closed_system_note': 'Do not reference specific wage '
                                                                      'negotiations.'},
 'cbs_Energy_Prices_Consumers_Electricity_Variable.csv': {'topic_category': 'energy',
                                                          'unit_description': 'variable electricity price '
                                                                              'for consumers (index or price '
                                                                              'level)',
                                                          'value_nature': 'index',
                                                          'higher_means': 'worse',
                                                          'direction_note': 'Rising values indicate higher '
                                                                            'electricity prices for '
                                                                            'consumers.',
                                                          'closed_system_note': 'IMPORTANT HIGH-RISK TOPIC: '
                                                                                'Electricity prices are '
                                                                                'heavily driven by external '
                                                                                'supply shocks and '
                                                                                'regulatory changes. '
                                                                                'Generate only abstract '
                                                                                'structural descriptions of '
                                                                                'the price pattern itself. '
                                                                                'Do not infer causes.'},
 'cbs_Inflation_CPI_Energy.csv': {'topic_category': 'inflation',
                                  'unit_description': 'consumer price index, energy category (base year = '
                                                      '100)',
                                  'value_nature': 'index',
                                  'higher_means': 'neutral',
                                  'direction_note': 'This is a PRICE INDEX for energy costs only. Rising '
                                                    'values indicate rising energy prices.',
                                  'closed_system_note': 'IMPORTANT HIGH-RISK TOPIC: Energy CPI is heavily '
                                                        'driven by external supply shocks. Generate only '
                                                        'abstract structural descriptions.'},
 'cbs_Terms_of_Trade_Goods_Export_Prices.csv': {'topic_category': 'trade',
                                                'unit_description': 'export price index for goods (base year '
                                                                    '= 100)',
                                                'value_nature': 'index',
                                                'higher_means': 'neutral',
                                                'direction_note': 'Rising values indicate rising export '
                                                                  'prices.',
                                                'closed_system_note': 'Interpret as structural price '
                                                                      'dynamics only. Do not reference '
                                                                      'specific commodity markets or trading '
                                                                      'partners.'},
 'cbs_Terms_of_Trade_Goods_Import_Prices.csv': {'topic_category': 'trade',
                                                'unit_description': 'import price index for goods (base year '
                                                                    '= 100)',
                                                'value_nature': 'index',
                                                'higher_means': 'neutral',
                                                'direction_note': 'Rising values indicate rising import '
                                                                  'prices.',
                                                'closed_system_note': 'Interpret as structural price '
                                                                      'dynamics only. Do not reference '
                                                                      'specific commodity markets or trading '
                                                                      'partners.'},
 'cbs_Terms_of_Trade_Goods_Terms_of_Trade.csv': {'topic_category': 'trade',
                                                 'unit_description': 'terms of trade index (export prices / '
                                                                     'import prices, base year = 100)',
                                                 'value_nature': 'index',
                                                 'higher_means': 'better',
                                                 'direction_note': 'Rising values mean export prices are '
                                                                   'rising faster than import prices — an '
                                                                   'improvement in trade competitiveness. '
                                                                   'Declining values indicate deteriorating '
                                                                   'terms of trade.',
                                                 'closed_system_note': 'Interpret as structural trade price '
                                                                       'dynamics only.'},
 'cbs_Labour_Participation_Unemployment_Total_15_24_Youth.csv': {'topic_category': 'labour_market',
                                                                 'unit_description': 'youth unemployment '
                                                                                     'rate, percentage of '
                                                                                     'labour force aged '
                                                                                     '15–24',
                                                                 'value_nature': 'rate_percentage',
                                                                 'higher_means': 'worse',
                                                                 'direction_note': 'This is a YOUTH '
                                                                                   'UNEMPLOYMENT RATE. '
                                                                                   'Declining values mean '
                                                                                   'fewer young people are '
                                                                                   'unemployed — an '
                                                                                   'improvement.',
                                                                 'closed_system_note': 'Interpret as '
                                                                                       'structural youth '
                                                                                       'labour market '
                                                                                       'dynamics only.'},
 'cbs_Inflation_CPI_Clothing.csv': {'topic_category': 'inflation',
                                    'unit_description': 'consumer price index, clothing and footwear (base '
                                                        'year = 100)',
                                    'value_nature': 'index',
                                    'higher_means': 'neutral',
                                    'direction_note': 'This is a PRICE INDEX for clothing and footwear only.',
                                    'closed_system_note': 'Do not reference specific seasonal or supply '
                                                          'chain events.'},
 'cbs_Inflation_CPI_Food.csv': {'topic_category': 'inflation',
                                'unit_description': 'consumer price index, food and non-alcoholic beverages '
                                                    '(base year = 100)',
                                'value_nature': 'index',
                                'higher_means': 'neutral',
                                'direction_note': 'This is a PRICE INDEX for food costs only. Rising values '
                                                  'indicate rising food prices.',
                                'closed_system_note': 'Interpret as structural price dynamics only. Do not '
                                                      'reference specific agricultural or supply events.'},
 'cbs_Inflation_CPI_Recreation.csv': {'topic_category': 'inflation',
                                      'unit_description': 'consumer price index, recreation and culture '
                                                          '(base year = 100)',
                                      'value_nature': 'index',
                                      'higher_means': 'neutral',
                                      'direction_note': 'This is a PRICE INDEX for recreation and culture '
                                                        'spending only.',
                                      'closed_system_note': 'Do not reference specific cultural or leisure '
                                                            'events.'},
 'cbs_Inflation_CPI_Transport.csv': {'topic_category': 'inflation',
                                     'unit_description': 'consumer price index, transport (base year = 100)',
                                     'value_nature': 'index',
                                     'higher_means': 'neutral',
                                     'direction_note': 'This is a PRICE INDEX for transport costs only.',
                                     'closed_system_note': 'Interpret as structural price dynamics only.'},
 'cbs_PPI_Output_2021_Manufacturing.csv': {'topic_category': 'producer_prices',
                                           'unit_description': 'producer price index, manufacturing output '
                                                               '(base year 2021 = 100)',
                                           'value_nature': 'index',
                                           'higher_means': 'neutral',
                                           'direction_note': 'This is a PRODUCER PRICE INDEX for '
                                                             'manufacturing. Rising values indicate rising '
                                                             'output prices at the producer level.',
                                           'closed_system_note': 'Do not reference specific input cost '
                                                                 'events.'},
 'cbs_PPI_Output_2021_Total_Industry.csv': {'topic_category': 'producer_prices',
                                            'unit_description': 'producer price index, total industry output '
                                                                '(base year 2021 = 100)',
                                            'value_nature': 'index',
                                            'higher_means': 'neutral',
                                            'direction_note': 'This is a PRODUCER PRICE INDEX for total '
                                                              'industry output.',
                                            'closed_system_note': 'Do not reference specific input cost '
                                                                  'events.'},
 'cbs_Producer_Price_Index_Manufacturing_Food_Products.csv': {'topic_category': 'producer_prices',
                                                              'unit_description': 'producer price index, '
                                                                                  'food products '
                                                                                  'manufacturing (base year '
                                                                                  '= 100)',
                                                              'value_nature': 'index',
                                                              'higher_means': 'neutral',
                                                              'direction_note': 'This is a PRODUCER PRICE '
                                                                                'INDEX for food product '
                                                                                'manufacturing.',
                                                              'closed_system_note': 'Interpret as structural '
                                                                                    'price dynamics only.'},
 'cbs_Producer_Price_Index_Manufacturing_Manufacturing.csv': {'topic_category': 'producer_prices',
                                                              'unit_description': 'producer price index, '
                                                                                  'total manufacturing (base '
                                                                                  'year = 100)',
                                                              'value_nature': 'index',
                                                              'higher_means': 'neutral',
                                                              'direction_note': 'This is a PRODUCER PRICE '
                                                                                'INDEX for total '
                                                                                'manufacturing.',
                                                              'closed_system_note': 'Interpret as structural '
                                                                                    'price dynamics only.'},
 'cbs_Producer_Price_Index_Manufacturing_Total_Industry.csv': {'topic_category': 'producer_prices',
                                                               'unit_description': 'producer price index, '
                                                                                   'total industry (base '
                                                                                   'year = 100)',
                                                               'value_nature': 'index',
                                                               'higher_means': 'neutral',
                                                               'direction_note': 'This is a PRODUCER PRICE '
                                                                                 'INDEX for total industry.',
                                                               'closed_system_note': 'Interpret as '
                                                                                     'structural price '
                                                                                     'dynamics only.'},
 'cbs_Producer_Confidence_Chemicals.csv': {'topic_category': 'producer_confidence',
                                           'unit_description': 'producer confidence index, chemicals sector '
                                                               '(points; positive = optimistic)',
                                           'value_nature': 'sentiment_index',
                                           'higher_means': 'better',
                                           'direction_note': 'Positive values indicate net optimism among '
                                                             'chemical sector producers; negative values '
                                                             'indicate net pessimism.',
                                           'closed_system_note': 'Interpret as structural sentiment dynamics '
                                                                 'within the sector.'},
 'cbs_Producer_Confidence_Food_Beverages.csv': {'topic_category': 'producer_confidence',
                                                'unit_description': 'producer confidence index, food and '
                                                                    'beverages sector (points; positive = '
                                                                    'optimistic)',
                                                'value_nature': 'sentiment_index',
                                                'higher_means': 'better',
                                                'direction_note': 'Positive values indicate net optimism '
                                                                  'among food and beverage producers.',
                                                'closed_system_note': 'Interpret as structural sentiment '
                                                                      'dynamics within the sector.'},
 'cbs_Producer_Confidence_Machinery.csv': {'topic_category': 'producer_confidence',
                                           'unit_description': 'producer confidence index, machinery sector '
                                                               '(points; positive = optimistic)',
                                           'value_nature': 'sentiment_index',
                                           'higher_means': 'better',
                                           'direction_note': 'Positive values indicate net optimism among '
                                                             'machinery sector producers.',
                                           'closed_system_note': 'Interpret as structural sentiment dynamics '
                                                                 'within the sector.'},
 'cbs_Producer_Confidence_Manufacturing_Total.csv': {'topic_category': 'producer_confidence',
                                                     'unit_description': 'producer confidence index, total '
                                                                         'manufacturing (points; positive = '
                                                                         'optimistic)',
                                                     'value_nature': 'sentiment_index',
                                                     'higher_means': 'better',
                                                     'direction_note': 'Positive values indicate net '
                                                                       'optimism across manufacturing '
                                                                       'producers.',
                                                     'closed_system_note': 'Interpret as structural '
                                                                           'sentiment dynamics.'},
 'cbs_Consumer_Confidence_Economic_Climate.csv': {'topic_category': 'consumer_confidence',
                                                  'unit_description': 'consumer assessment of economic '
                                                                      'climate (points; positive = '
                                                                      'optimistic)',
                                                  'value_nature': 'sentiment_index',
                                                  'higher_means': 'better',
                                                  'direction_note': 'Positive values indicate consumers view '
                                                                    'the economic climate favourably.',
                                                  'closed_system_note': 'Interpret as structural sentiment '
                                                                        'dynamics only.'},
 'cbs_Consumer_Confidence_Econ_Situation_Last12M.csv': {'topic_category': 'consumer_confidence',
                                                        'unit_description': 'consumer assessment of economic '
                                                                            'situation over the past 12 '
                                                                            'months (points)',
                                                        'value_nature': 'sentiment_index',
                                                        'higher_means': 'better',
                                                        'direction_note': 'Rising values indicate consumers '
                                                                          'report improving retrospective '
                                                                          'economic conditions.',
                                                        'closed_system_note': 'Interpret as structural '
                                                                              'sentiment dynamics only.'},
 'cbs_Consumer_Confidence_Econ_Situation_Next12M.csv': {'topic_category': 'consumer_confidence',
                                                        'unit_description': 'consumer expectations for '
                                                                            'economic situation over the '
                                                                            'next 12 months (points)',
                                                        'value_nature': 'sentiment_index',
                                                        'higher_means': 'better',
                                                        'direction_note': 'Rising values indicate consumers '
                                                                          'expect improving economic '
                                                                          'conditions ahead.',
                                                        'closed_system_note': 'Interpret as structural '
                                                                              'forward-looking sentiment '
                                                                              'dynamics only.'},
 'cbs_Consumer_Confidence_Financial_Situation_Last12M.csv': {'topic_category': 'consumer_confidence',
                                                             'unit_description': 'consumer assessment of '
                                                                                 'personal financial '
                                                                                 'situation over past 12 '
                                                                                 'months (points)',
                                                             'value_nature': 'sentiment_index',
                                                             'higher_means': 'better',
                                                             'direction_note': 'Rising values indicate '
                                                                               'consumers report improving '
                                                                               'personal finances '
                                                                               'retrospectively.',
                                                             'closed_system_note': 'Interpret as structural '
                                                                                   'sentiment dynamics '
                                                                                   'only.'},
 'cbs_Consumer_Confidence_Financial_Situation_Next12M.csv': {'topic_category': 'consumer_confidence',
                                                             'unit_description': 'consumer expectations for '
                                                                                 'personal financial '
                                                                                 'situation over next 12 '
                                                                                 'months (points)',
                                                             'value_nature': 'sentiment_index',
                                                             'higher_means': 'better',
                                                             'direction_note': 'Rising values indicate '
                                                                               'consumers expect their '
                                                                               'personal finances to '
                                                                               'improve.',
                                                             'closed_system_note': 'Interpret as structural '
                                                                                   'forward-looking '
                                                                                   'sentiment dynamics '
                                                                                   'only.'},
 'cbs_Consumer_Confidence_Willingness_Buy.csv': {'topic_category': 'consumer_confidence',
                                                 'unit_description': 'consumer willingness to make major '
                                                                     'purchases (points; positive = willing)',
                                                 'value_nature': 'sentiment_index',
                                                 'higher_means': 'better',
                                                 'direction_note': 'Rising values indicate consumers are '
                                                                   'more willing to make large purchases.',
                                                 'closed_system_note': 'Interpret as structural sentiment '
                                                                       'dynamics only.'},
 'cbs_Civil_Engineering_Input_Price_Index_Railways.csv': {'topic_category': 'construction',
                                                          'unit_description': 'input price index for railway '
                                                                              'civil engineering (base year '
                                                                              '= 100)',
                                                          'value_nature': 'index',
                                                          'higher_means': 'neutral',
                                                          'direction_note': 'Rising values indicate rising '
                                                                            'input costs for railway '
                                                                            'construction projects.',
                                                          'closed_system_note': 'Interpret as structural '
                                                                                'cost dynamics only.'},
 'cbs_Civil_Engineering_Input_Price_Index_Road_Construction_Asphalt.csv': {'topic_category': 'construction',
                                                                           'unit_description': 'input price '
                                                                                               'index for '
                                                                                               'road '
                                                                                               'construction '
                                                                                               'using '
                                                                                               'asphalt '
                                                                                               '(base year = '
                                                                                               '100)',
                                                                           'value_nature': 'index',
                                                                           'higher_means': 'neutral',
                                                                           'direction_note': 'Rising values '
                                                                                             'indicate '
                                                                                             'rising input '
                                                                                             'costs for '
                                                                                             'asphalt road '
                                                                                             'construction.',
                                                                           'closed_system_note': 'Interpret '
                                                                                                 'as '
                                                                                                 'structural '
                                                                                                 'cost '
                                                                                                 'dynamics '
                                                                                                 'only.'},
 'cbs_Civil_Engineering_Input_Price_Index_Road_Construction_Brick.csv': {'topic_category': 'construction',
                                                                         'unit_description': 'input price '
                                                                                             'index for road '
                                                                                             'construction '
                                                                                             'using brick '
                                                                                             '(base year = '
                                                                                             '100)',
                                                                         'value_nature': 'index',
                                                                         'higher_means': 'neutral',
                                                                         'direction_note': 'Rising values '
                                                                                           'indicate rising '
                                                                                           'input costs for '
                                                                                           'brick road '
                                                                                           'construction.',
                                                                         'closed_system_note': 'Interpret as '
                                                                                               'structural '
                                                                                               'cost '
                                                                                               'dynamics '
                                                                                               'only.'},
 'cbs_Civil_Engineering_Input_Price_Index_Total_Civil_Engineering.csv': {'topic_category': 'construction',
                                                                         'unit_description': 'input price '
                                                                                             'index for '
                                                                                             'total civil '
                                                                                             'engineering '
                                                                                             '(base year = '
                                                                                             '100)',
                                                                         'value_nature': 'index',
                                                                         'higher_means': 'neutral',
                                                                         'direction_note': 'Rising values '
                                                                                           'indicate rising '
                                                                                           'input costs '
                                                                                           'across civil '
                                                                                           'engineering.',
                                                                         'closed_system_note': 'Interpret as '
                                                                                               'structural '
                                                                                               'cost '
                                                                                               'dynamics '
                                                                                               'only.'},
 'cbs_Construction_Production_Index_Civil_Engineering.csv': {'topic_category': 'construction',
                                                             'unit_description': 'production volume index '
                                                                                 'for civil engineering '
                                                                                 '(base year = 100)',
                                                             'value_nature': 'index',
                                                             'higher_means': 'better',
                                                             'direction_note': 'Rising values indicate '
                                                                               'increasing civil engineering '
                                                                               'production volume.',
                                                             'closed_system_note': 'Interpret as structural '
                                                                                   'production dynamics '
                                                                                   'only.'},
 'cbs_Construction_Production_Index_Construction_Buildings.csv': {'topic_category': 'construction',
                                                                  'unit_description': 'production volume '
                                                                                      'index for building '
                                                                                      'construction (base '
                                                                                      'year = 100)',
                                                                  'value_nature': 'index',
                                                                  'higher_means': 'better',
                                                                  'direction_note': 'Rising values indicate '
                                                                                    'increasing building '
                                                                                    'construction volume.',
                                                                  'closed_system_note': 'Interpret as '
                                                                                        'structural '
                                                                                        'production dynamics '
                                                                                        'only.'},
 'cbs_Construction_Production_Index_Total_Construction.csv': {'topic_category': 'construction',
                                                              'unit_description': 'production volume index '
                                                                                  'for total construction '
                                                                                  '(base year = 100)',
                                                              'value_nature': 'index',
                                                              'higher_means': 'better',
                                                              'direction_note': 'Rising values indicate '
                                                                                'increasing total '
                                                                                'construction volume.',
                                                              'closed_system_note': 'Interpret as structural '
                                                                                    'production dynamics '
                                                                                    'only.'},
 'cbs_Existing_Homes_Prices_Sales_AvgPrice_Average_Price.csv': {'topic_category': 'housing',
                                                                'unit_description': 'average sale price of '
                                                                                    'existing dwellings '
                                                                                    '(euros per dwelling)',
                                                                'value_nature': 'price_level',
                                                                'higher_means': 'neutral',
                                                                'direction_note': 'Rising values indicate '
                                                                                  'rising average house '
                                                                                  'prices.',
                                                                'closed_system_note': 'Interpret as '
                                                                                      'structural housing '
                                                                                      'market price dynamics '
                                                                                      'only.'},
 'cbs_Existing_Homes_Prices_Sales_AvgPrice_Price_Index.csv': {'topic_category': 'housing',
                                                              'unit_description': 'price index for existing '
                                                                                  'dwellings (base year = '
                                                                                  '100)',
                                                              'value_nature': 'index',
                                                              'higher_means': 'neutral',
                                                              'direction_note': 'This is a PRICE INDEX for '
                                                                                'existing homes. Rising '
                                                                                'values indicate house '
                                                                                'prices are rising relative '
                                                                                'to the base year.',
                                                              'closed_system_note': 'Interpret as structural '
                                                                                    'housing market price '
                                                                                    'dynamics only.'},
 'cbs_Household_Consumption_Expenditure_Goods.csv': {'topic_category': 'consumption',
                                                     'unit_description': 'household consumption expenditure '
                                                                         'on goods (volume index, base year '
                                                                         '= 100)',
                                                     'value_nature': 'index',
                                                     'higher_means': 'better',
                                                     'direction_note': 'Rising values indicate increasing '
                                                                       'household spending on goods in real '
                                                                       'terms.',
                                                     'closed_system_note': 'Interpret as structural '
                                                                           'consumption dynamics only.'},
 'cbs_Household_Consumption_Expenditure_Services.csv': {'topic_category': 'consumption',
                                                        'unit_description': 'household consumption '
                                                                            'expenditure on services (volume '
                                                                            'index, base year = 100)',
                                                        'value_nature': 'index',
                                                        'higher_means': 'better',
                                                        'direction_note': 'Rising values indicate increasing '
                                                                          'household spending on services in '
                                                                          'real terms.',
                                                        'closed_system_note': 'Interpret as structural '
                                                                              'consumption dynamics only.'},
 'cbs_Household_Consumption_Expenditure_Total_Consumption.csv': {'topic_category': 'consumption',
                                                                 'unit_description': 'total household '
                                                                                     'consumption '
                                                                                     'expenditure (volume '
                                                                                     'index, base year = '
                                                                                     '100)',
                                                                 'value_nature': 'index',
                                                                 'higher_means': 'better',
                                                                 'direction_note': 'Rising values indicate '
                                                                                   'increasing total '
                                                                                   'household spending in '
                                                                                   'real terms.',
                                                                 'closed_system_note': 'Interpret as '
                                                                                       'structural '
                                                                                       'consumption dynamics '
                                                                                       'only.'},
 'cbs_Industry_Production_Sales_Food_Products.csv': {'topic_category': 'industrial_production',
                                                     'unit_description': 'production/sales index for food '
                                                                         'products industry (base year = '
                                                                         '100)',
                                                     'value_nature': 'index',
                                                     'higher_means': 'better',
                                                     'direction_note': 'Rising values indicate increasing '
                                                                       'food product industry output or '
                                                                       'sales.',
                                                     'closed_system_note': 'Interpret as structural '
                                                                           'industrial dynamics only.'},
 'cbs_Industry_Production_Sales_Manufacturing.csv': {'topic_category': 'industrial_production',
                                                     'unit_description': 'production/sales index for total '
                                                                         'manufacturing (base year = 100)',
                                                     'value_nature': 'index',
                                                     'higher_means': 'better',
                                                     'direction_note': 'Rising values indicate increasing '
                                                                       'manufacturing output or sales.',
                                                     'closed_system_note': 'Interpret as structural '
                                                                           'industrial dynamics only.'},
 'cbs_Investment_Tangible_Assets_Volume_Volume_Index.csv': {'topic_category': 'investment',
                                                            'unit_description': 'volume index of investment '
                                                                                'in tangible fixed assets '
                                                                                '(base year = 100)',
                                                            'value_nature': 'index',
                                                            'higher_means': 'better',
                                                            'direction_note': 'Rising values indicate '
                                                                              'increasing real investment in '
                                                                              'physical capital.',
                                                            'closed_system_note': 'Interpret as structural '
                                                                                  'investment dynamics '
                                                                                  'only.'},
 'cbs_Retail_Turnover_Manufacturing.csv': {'topic_category': 'retail',
                                           'unit_description': 'retail turnover index for '
                                                               'manufacturing-related retail (base year = '
                                                               '100)',
                                           'value_nature': 'index',
                                           'higher_means': 'better',
                                           'direction_note': 'Rising values indicate increasing retail '
                                                             'turnover in this category.',
                                           'closed_system_note': 'Interpret as structural retail dynamics '
                                                                 'only.'},
 'cbs_Consumer_Confidence_Confidence_Index.csv': {'topic_category': 'consumer_confidence',
                                                  'unit_description': 'consumer confidence index (points; '
                                                                      'positive = optimistic, negative = '
                                                                      'pessimistic)',
                                                  'value_nature': 'sentiment_index',
                                                  'higher_means': 'better',
                                                  'direction_note': 'IMPORTANT: The zero line is the '
                                                                    'neutrality threshold. Positive values '
                                                                    'indicate net optimism; negative values '
                                                                    'indicate net pessimism. The index can '
                                                                    'cross zero.',
                                                  'closed_system_note': 'Interpret as structural sentiment '
                                                                        'dynamics. Do not reference specific '
                                                                        'economic events.'},
 'cbs_Existing_Homes_Prices_Sales_AvgPrice_Sold_Dwellings.csv': {'topic_category': 'housing',
                                                                 'unit_description': 'average sale price of '
                                                                                     'existing dwellings '
                                                                                     '(euros; 12-month '
                                                                                     'rolling sum)',
                                                                 'value_nature': 'flow_value',
                                                                 'higher_means': 'neutral',
                                                                 'direction_note': 'This represents the '
                                                                                   'rolling total '
                                                                                   'transaction value, not '
                                                                                   'individual house prices.',
                                                                 'closed_system_note': 'Interpret as '
                                                                                       'structural housing '
                                                                                       'market dynamics '
                                                                                       'only.'},
 'cbs_Population_Dynamics_Monthly_Deaths.csv': {'topic_category': 'population',
                                                'unit_description': 'number of deaths (12-month rolling sum)',
                                                'value_nature': 'flow_count',
                                                'higher_means': 'worse',
                                                'direction_note': 'Rising values indicate increasing '
                                                                  'mortality. This is a demographic flow '
                                                                  'measure.',
                                                'closed_system_note': 'IMPORTANT: This topic has strong '
                                                                      'external drivers. Generate only '
                                                                      'abstract structural descriptions '
                                                                      'derivable from the pattern itself. Do '
                                                                      'not infer causes.'},
 'cbs_Population_Dynamics_Monthly_Live_Births.csv': {'topic_category': 'population',
                                                     'unit_description': 'number of live births (12-month '
                                                                         'rolling sum)',
                                                     'value_nature': 'flow_count',
                                                     'higher_means': 'neutral',
                                                     'direction_note': 'Rising values indicate increasing '
                                                                       'birth rates. This is a demographic '
                                                                       'flow measure.',
                                                     'closed_system_note': 'Generate only abstract '
                                                                           'structural descriptions. Do not '
                                                                           'infer causes.'},
 'cbs_Population_Dynamics_Monthly_Net_Migration.csv': {'topic_category': 'population',
                                                       'unit_description': 'net migration (12-month rolling '
                                                                           'sum of arrivals minus '
                                                                           'departures)',
                                                       'value_nature': 'flow_count',
                                                       'higher_means': 'neutral',
                                                       'direction_note': 'IMPORTANT: Each data point is a '
                                                                         '12-MONTH ROLLING SUM, not a '
                                                                         'cumulative total since the series '
                                                                         'start. Rising values mean net '
                                                                         'inflow increased; declining values '
                                                                         'mean net inflow decreased or '
                                                                         'outflow grew.',
                                                       'closed_system_note': 'IMPORTANT: This topic has '
                                                                             'strong external drivers. '
                                                                             'Generate only abstract '
                                                                             'structural descriptions.'}}

DEFAULT_SEMANTICS = {
    "topic_category": "economic_indicator",
    "unit_description": "economic measure",
    "value_nature": "level_or_index",
    "higher_means": "depends_on_context",
    "direction_note": "Interpret direction based on the variable itself. Avoid unsupported domain assumptions.",
    "closed_system_note": "Stay within what can be inferred from the chart structure and variable semantics only.",
}


# ----------------------------------------------------------------
# High-risk topics
# These indicators are prone to unsupported external-event attribution.
# ----------------------------------------------------------------

HIGH_RISK_TOPICS = {
    "cbs_Inflation_CPI_Energy.csv",
    "cbs_Population_Dynamics_Monthly_Deaths.csv",
    "cbs_Population_Dynamics_Monthly_Live_Births.csv",
    "cbs_Population_Dynamics_Monthly_Net_Migration.csv",
}

# These files use rolling sums in a_04 instead of rolling means.
ROLLING_SUM_FILES = {
    "cbs_Bankruptcies_Key_Figures_Companies.csv",
    "cbs_Bankruptcies_Key_Figures_Individuals.csv",
    "cbs_Bankruptcies_Key_Figures_Total.csv",
    "cbs_Population_Dynamics_Monthly_Deaths.csv",
    "cbs_Population_Dynamics_Monthly_Live_Births.csv",
    "cbs_Population_Dynamics_Monthly_Net_Migration.csv",
}

def load_semantic_pool_ids(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        ids = []
        for i, rec in enumerate(data):
            if not isinstance(rec, dict):
                continue
            sid = rec.get("id") or rec.get("sample_id") or rec.get("image_id")
            if not sid:
                img = rec.get("image")
                if isinstance(img, str):
                    sid = Path(img).name
            if not sid:
                raise ValueError(f"Cannot identify sample id for semantic_pool record {i}")
            ids.append(sid)
        return ids

    if isinstance(data, dict):
        if all(isinstance(v, dict) for v in data.values()):
            return list(data.keys())

        for k in ["samples", "items", "records", "data"]:
            if k in data and isinstance(data[k], list):
                ids = []
                for i, rec in enumerate(data[k]):
                    if not isinstance(rec, dict):
                        continue
                    sid = rec.get("id") or rec.get("sample_id") or rec.get("image_id")
                    if not sid:
                        img = rec.get("image")
                        if isinstance(img, str):
                            sid = Path(img).name
                    if not sid:
                        raise ValueError(f"Cannot identify sample id for semantic_pool[{k}][{i}]")
                    ids.append(sid)
                return ids

    raise ValueError("Unsupported semantic_pool JSON structure")


# ----------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------

def parse_date(s: str) -> pd.Timestamp:
    return pd.to_datetime(s)

def to_float(v):
    try:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            if np.isnan(v):
                return None
            return float(v)
        return float(str(v).replace(",", ""))
    except Exception:
        return None

def approx_equal(a, b, tol=1e-6):
    if a is None or b is None:
        return False
    return abs(a - b) <= tol

def months_between(start: str, end: str) -> int:
    ds = parse_date(start)
    de = parse_date(end)
    return (de.year - ds.year) * 12 + (de.month - ds.month) + 1


def fill_zero_signs(signs: np.ndarray) -> np.ndarray:
    signs = signs.astype(float).copy()
    if len(signs) == 0:
        return signs

    last = 0
    for i in range(len(signs)):
        if signs[i] == 0:
            signs[i] = last
        else:
            last = signs[i]

    nxt = 0
    for i in range(len(signs) - 1, -1, -1):
        if signs[i] == 0:
            signs[i] = nxt
        else:
            nxt = signs[i]
    return signs


def compute_lag_autocorr(values: np.ndarray, lag: int = 12) -> Optional[float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < lag * 2:
        return None
    x = arr[:-lag]
    y = arr[lag:]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def count_zero_crossings(values: np.ndarray, threshold: float = 0.0) -> int:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return 0
    centered = arr - threshold
    signs = np.sign(centered)
    signs = fill_zero_signs(signs)
    crossings = 0
    prev = signs[0]
    for s in signs[1:]:
        if prev != 0 and s != 0 and s != prev:
            crossings += 1
        if s != 0:
            prev = s
    return int(crossings)


def is_sentiment_like(csv_source: str, semantics: dict) -> bool:
    value_nature = str(semantics.get("value_nature", "")).lower()
    source_lower = csv_source.lower()
    return (
        value_nature == "sentiment_index"
        or "confidence" in source_lower
    )


def compute_max_internal_prominence_ratio(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 5:
        return 0.0
    total_range = float(np.max(arr) - np.min(arr))
    if total_range <= 0:
        return 0.0

    best = 0.0
    for i in range(1, n - 1):
        prev_v, cur_v, next_v = arr[i - 1], arr[i], arr[i + 1]
        is_peak = cur_v > prev_v and cur_v > next_v
        is_trough = cur_v < prev_v and cur_v < next_v
        if not (is_peak or is_trough):
            continue

        if is_peak:
            left_support = float(np.min(arr[:i]))
            right_support = float(np.min(arr[i + 1:]))
            prom = cur_v - max(left_support, right_support)
        else:
            left_support = float(np.max(arr[:i]))
            right_support = float(np.max(arr[i + 1:]))
            prom = min(left_support, right_support) - cur_v

        best = max(best, float(prom) / total_range)
    return round(float(best), 4)


def compute_slope_regime_ratio(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 3:
        return 1.0

    diffs = np.abs(np.diff(arr))
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) == 0:
        return 1.0

    total_range = float(np.max(arr) - np.min(arr)) if n >= 2 else 0.0
    eps = max(1e-8, total_range * 0.01)
    active = diffs[diffs > eps]
    if len(active) < 3:
        active = diffs[diffs > 0]
    if len(active) < 3:
        return 1.0

    p10 = float(np.percentile(active, 10))
    p90 = float(np.percentile(active, 90))
    if p10 <= 1e-8:
        return 99.0
    return round(float(min(99.0, p90 / p10)), 4)


def compute_series_metrics(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)

    if n < 2:
        return {
            "n": int(n),
            "path_efficiency": 1.0,
            "turning_density": 0.0,
            "prominence_ratio": 0.0,
            "shock_ratio": 0.0,
            "slope_regime_ratio": 1.0,
            "total_range": 0.0,
        }

    diffs = np.diff(arr)
    total_path = float(np.sum(np.abs(diffs)))
    total_range = float(np.max(arr) - np.min(arr))
    net_change = float(arr[-1] - arr[0])

    path_efficiency = 1.0 if total_path == 0 else min(1.0, abs(net_change) / total_path)

    signs = fill_zero_signs(np.sign(diffs))
    sign_flips = 0
    prev = None
    for s in signs:
        if s == 0:
            continue
        if prev is not None and s != prev:
            sign_flips += 1
        prev = s
    turning_density = sign_flips / max(1, n - 2)

    prominence_ratio = compute_max_internal_prominence_ratio(arr)
    shock_ratio = 0.0 if total_range <= 0 else float(np.max(np.abs(diffs))) / total_range
    slope_regime_ratio = compute_slope_regime_ratio(arr)

    return {
        "n": int(n),
        "path_efficiency": round(float(path_efficiency), 4),
        "turning_density": round(float(turning_density), 4),
        "prominence_ratio": round(float(prominence_ratio), 4),
        "shock_ratio": round(float(shock_ratio), 4),
        "slope_regime_ratio": round(float(slope_regime_ratio), 4),
        "total_range": round(float(total_range), 4),
    }


def classify_morphology_family(
    csv_source: str,
    semantics: dict,
    analysis_values: np.ndarray,
    raw_values: np.ndarray,
) -> dict:
    analysis_metrics = compute_series_metrics(analysis_values)
    raw_arr = np.asarray(raw_values, dtype=float)
    raw_arr = raw_arr[np.isfinite(raw_arr)]

    lag12_acf_raw = compute_lag_autocorr(raw_arr, lag=12)
    lag12_for_gate = lag12_acf_raw if lag12_acf_raw is not None else 0.0
    sentiment_like = is_sentiment_like(csv_source, semantics)
    zero_cross_count_raw = count_zero_crossings(raw_arr, threshold=0.0) if sentiment_like else 0

    pe = analysis_metrics["path_efficiency"]
    td = analysis_metrics["turning_density"]
    pr = analysis_metrics["prominence_ratio"]
    sr = analysis_metrics["shock_ratio"]
    srr = analysis_metrics["slope_regime_ratio"]

    seasonal_memory_flag = (lag12_for_gate >= 0.75 and td >= 0.14)
    sentiment_zero_crossing = sentiment_like and zero_cross_count_raw >= 1
    shock_like = sr >= 0.22

    if (
        pe >= 0.72
        and td <= 0.08
        and pr <= 0.18
        and sr <= 0.22
        and not seasonal_memory_flag
        and (not sentiment_like or zero_cross_count_raw == 0)
    ):
        family = "trend_drift"
        family_confidence = "high" if (pe >= 0.82 and td <= 0.05 and pr <= 0.12) else "medium"
    elif (
        (sentiment_like and (td >= 0.16 or zero_cross_count_raw >= 1))
        or (td >= 0.18 and pe <= 0.18 and (lag12_for_gate >= 0.72 or pr >= 0.45))
        or (lag12_for_gate >= 0.80 and td >= 0.14 and pe <= 0.25)
    ):
        family = "oscillatory_or_seasonal_regime"
        family_confidence = "high" if (td >= 0.22 or lag12_for_gate >= 0.85 or zero_cross_count_raw >= 2) else "medium"
    else:
        family = "structural_swing"
        family_confidence = "high" if (pr >= 0.28 or sr >= 0.22 or (0.18 <= td <= 0.30)) else "medium"

    return {
        "family": family,
        "family_confidence": family_confidence,
        "gate_metrics": {
            "path_efficiency": pe,
            "turning_density": td,
            "prominence_ratio": pr,
            "shock_ratio": sr,
            "slope_regime_ratio": srr,
            "lag12_acf_raw": round(float(lag12_acf_raw), 4) if lag12_acf_raw is not None else None,
            "zero_cross_count_raw": int(zero_cross_count_raw),
        },
        "modifier_flags": {
            "seasonal_memory": bool(seasonal_memory_flag),
            "sentiment_zero_crossing": bool(sentiment_zero_crossing),
            "shock_like": bool(shock_like),
            "sentiment_like": bool(sentiment_like),
        },
    }


# ----------------------------------------------------------------
# Trend phase detection
# ----------------------------------------------------------------

def detect_trend_phases(
    values: np.ndarray,
    dates: pd.DatetimeIndex,
    min_phase_months: int = 4,
    smoothing_window: int = 3,
) -> Tuple[str, List[dict], List[dict]]:
    """
    Detect trend phases in a time series.

    Returns:
      trend_type   : "monotonic_increasing" / "monotonic_decreasing" / "non_monotonic"
      phases       : direction, boundaries, and magnitude for each phase
      inflections  : local extrema
    """
    n = len(values)

    if n < 2:
        return (
            "monotonic_increasing",
            [{
                "direction": "increasing",
                "from_date": str(dates[0])[:10],
                "to_date": str(dates[-1])[:10],
                "start_value": round(float(values[0]), 4),
                "end_value": round(float(values[-1]), 4),
                "magnitude": round(float(values[-1] - values[0]), 4),
            }],
            [],
        )

    if n < 6:
        direction = "increasing" if values[-1] >= values[0] else "decreasing"
        return (
            f"monotonic_{direction}",
            [{
                "direction": direction,
                "from_date": str(dates[0])[:10],
                "to_date": str(dates[-1])[:10],
                "start_value": round(float(values[0]), 4),
                "end_value": round(float(values[-1]), 4),
                "magnitude": round(float(values[-1] - values[0]), 4),
            }],
            [],
        )

    sw = min(smoothing_window, max(2, n // 2))
    if sw > 1:
        smoothed = pd.Series(values).rolling(sw, center=True, min_periods=1).mean().to_numpy()
    else:
        smoothed = values.copy()

    diffs = np.diff(smoothed)
    signs = fill_zero_signs(np.sign(diffs))

    def find_direction_changes(signs_arr):
        changes = []
        non_zero = signs_arr[signs_arr != 0]
        if len(non_zero) == 0:
            return changes
        current = non_zero[0]
        for i, s in enumerate(signs_arr):
            if s != 0 and s != current:
                changes.append(i + 1)
                current = s
        return changes

    change_pts = find_direction_changes(signs)

    boundaries = [0] + change_pts + [n - 1]
    raw_phases = []
    for i in range(len(boundaries) - 1):
        s_idx = boundaries[i]
        e_idx = boundaries[i + 1]
        if e_idx <= s_idx:
            continue
        seg_vals = values[s_idx:e_idx + 1]
        magnitude = float(seg_vals[-1] - seg_vals[0])
        direction = "increasing" if magnitude >= 0 else "decreasing"
        raw_phases.append({
            "start_idx": s_idx,
            "end_idx": e_idx,
            "direction": direction,
            "start_value": round(float(seg_vals[0]), 4),
            "end_value": round(float(seg_vals[-1]), 4),
            "magnitude": round(magnitude, 4),
            "length": e_idx - s_idx,
        })

    if not raw_phases:
        direction = "increasing" if values[-1] >= values[0] else "decreasing"
        raw_phases = [{
            "start_idx": 0,
            "end_idx": n - 1,
            "direction": direction,
            "start_value": round(float(values[0]), 4),
            "end_value": round(float(values[-1]), 4),
            "magnitude": round(float(values[-1] - values[0]), 4),
            "length": n - 1,
        }]

    def merge_short_phases(phases, min_len):
        if len(phases) <= 1:
            return phases
        merged = [dict(phases[0])]
        for ph in phases[1:]:
            if ph["length"] < min_len:
                prev = merged[-1]
                prev["end_idx"] = ph["end_idx"]
                prev["end_value"] = ph["end_value"]
                prev["magnitude"] = round(prev["end_value"] - prev["start_value"], 4)
                prev["length"] = prev["end_idx"] - prev["start_idx"]
                prev["direction"] = "increasing" if prev["magnitude"] >= 0 else "decreasing"
            else:
                merged.append(dict(ph))
        return merged

    phases = merge_short_phases(raw_phases, min_phase_months)

    total_range = float(np.max(values) - np.min(values))
    significant_phases = phases
    if total_range > 0:
        significant_phases = [
            p for p in phases
            if abs(p["magnitude"]) / total_range > 0.15
        ]
        if not significant_phases:
            significant_phases = phases

    internal_prominent = False
    if total_range > 0 and n >= 5:
        for i in range(1, n - 1):
            prev_v, cur_v, next_v = values[i - 1], values[i], values[i + 1]
            is_peak = cur_v > prev_v and cur_v > next_v
            is_trough = cur_v < prev_v and cur_v < next_v
            if not (is_peak or is_trough):
                continue

            if is_peak:
                left_support = float(np.min(values[:i]))
                right_support = float(np.min(values[i + 1:]))
                prom = cur_v - max(left_support, right_support)
            else:
                left_support = float(np.max(values[:i]))
                right_support = float(np.max(values[i + 1:]))
                prom = min(left_support, right_support) - cur_v

            prom_ratio = prom / total_range
            if prom_ratio >= 0.12:
                internal_prominent = True
                break

    direction_set = {p["direction"] for p in significant_phases}

    if len(direction_set) >= 2:
        trend_type = "non_monotonic"
        final_phases = significant_phases
    elif internal_prominent and len(phases) >= 2:
        trend_type = "non_monotonic"
        final_phases = phases
    else:
        direction = "increasing" if values[-1] >= values[0] else "decreasing"
        trend_type = f"monotonic_{direction}"
        final_phases = [{
            "start_idx": 0,
            "end_idx": n - 1,
            "direction": direction,
            "start_value": round(float(values[0]), 4),
            "end_value": round(float(values[-1]), 4),
            "magnitude": round(float(values[-1] - values[0]), 4),
            "length": n - 1,
        }]

    inflections = []
    if trend_type == "non_monotonic":
        for i in range(1, len(final_phases)):
            prev_ph = final_phases[i - 1]
            turn_idx = prev_ph["end_idx"]
            inflection_type = (
                "local_peak" if prev_ph["direction"] == "increasing"
                else "local_trough"
            )
            inflections.append({
                "date": str(dates[turn_idx])[:10],
                "value": round(float(values[turn_idx]), 4),
                "type": inflection_type,
            })

    formatted_phases = []
    for ph in final_phases:
        formatted_phases.append({
            "direction": ph["direction"],
            "from_date": str(dates[ph["start_idx"]])[:10],
            "to_date": str(dates[ph["end_idx"]])[:10],
            "start_value": ph["start_value"],
            "end_value": ph["end_value"],
            "magnitude": ph["magnitude"],
        })

    return trend_type, formatted_phases, inflections


# ----------------------------------------------------------------
# Boundary flags
# ----------------------------------------------------------------

def compute_boundary_flags(v1_anchor: dict) -> dict:
    vals = v1_anchor.get("values", {})
    ext  = v1_anchor.get("extremes", {})

    start_v = to_float(vals.get("start_value"))
    end_v   = to_float(vals.get("end_value"))
    peak_v  = to_float(ext.get("peak", {}).get("value"))
    trough_v= to_float(ext.get("trough", {}).get("value"))

    return {
        "peak_at_window_start":   approx_equal(start_v, peak_v),
        "peak_at_window_end":     approx_equal(end_v, peak_v),
        "trough_at_window_start": approx_equal(start_v, trough_v),
        "trough_at_window_end":   approx_equal(end_v, trough_v),
    }


# ----------------------------------------------------------------
# Morphology gate + binary phase signal + final routing
# ----------------------------------------------------------------

def compute_binary_phase_signal(
    trend_type: str,
    phases: List[dict],
    morphology_signal: dict,
) -> dict:
    phase_count_raw = max(1, len(phases))
    prominent_internal_turn = morphology_signal["gate_metrics"]["prominence_ratio"] >= 0.28
    shock_like = morphology_signal["modifier_flags"]["shock_like"]
    reversal_evidence = bool(trend_type == "non_monotonic" or prominent_internal_turn or shock_like)

    eligible_for_phase_routing = morphology_signal["family"] != "oscillatory_or_seasonal_regime"
    binary_pc = "pc_gt1" if phase_count_raw > 1 else "pc1"

    return {
        "eligible_for_phase_routing": bool(eligible_for_phase_routing),
        "phase_count_raw": int(phase_count_raw),
        "binary_pc": binary_pc,
        "reversal_evidence": bool(reversal_evidence),
        "prominent_internal_turn": bool(prominent_internal_turn),
    }


def build_generation_guidance(route_label: str, family: str) -> str:
    if route_label == "simple":
        return (
            "SIMPLE PATTERN: This window is treated as structurally straightforward. "
            "P1 should describe the dominant direction and the endpoint relationship only. "
            "Do not invent hidden reversals, extra phases, or unsupported mechanisms. "
            "P2 should stay short and explain only the structural implication of the sustained movement."
        )

    if family == "oscillatory_or_seasonal_regime":
        family_note = (
            "This window contains repeated swings, cyclical structure, or regime-like oscillation. "
            "Do not collapse it into a single start-to-end summary."
        )
    elif family == "structural_swing":
        family_note = (
            "This window contains internal structure such as a reversal, plateau, shock, or step-like shift. "
            "Describe the internal pattern explicitly."
        )
    else:
        family_note = (
            "This window is routed to complex because the internal structure is not safely reducible to a simple monotonic description."
        )

    return (
        "COMPLEX PATTERN: " + family_note + " "
        "P1 must cover the relevant internal movement rather than only the final direction. "
        "P2 should explain the full trajectory within the closed system described by the variable semantics."
    )


def compute_complexity_signal(
    trend_type: str,
    duration_months: int,
    morphology_signal: dict,
    binary_phase_signal: dict,
) -> dict:
    family = morphology_signal["family"]
    family_confidence = morphology_signal.get("family_confidence", "low")
    pe = morphology_signal["gate_metrics"]["path_efficiency"]
    td = morphology_signal["gate_metrics"]["turning_density"]
    pr = morphology_signal["gate_metrics"]["prominence_ratio"]
    sr = morphology_signal["gate_metrics"]["shock_ratio"]
    srr = morphology_signal["gate_metrics"].get("slope_regime_ratio", 1.0)
    phase_count_raw = binary_phase_signal["phase_count_raw"]
    reversal_evidence = binary_phase_signal.get("reversal_evidence", False)
    prominent_internal_turn = binary_phase_signal.get("prominent_internal_turn", False)
    seasonal_memory = morphology_signal.get("modifier_flags", {}).get("seasonal_memory", False)
    sentiment_like = morphology_signal.get("modifier_flags", {}).get("sentiment_like", False)

    route_label = "complex"
    routing_reason = "default_complex"

    if family == "trend_drift":
        if (
            family_confidence == "high"
            and phase_count_raw <= 1
            and trend_type.startswith("monotonic")
            and pe >= 0.95
            and td <= 0.04
            and pr <= 0.05
            and sr <= 0.08
            and srr <= 1.8
            and not reversal_evidence
            and not prominent_internal_turn
            and not seasonal_memory
            and not sentiment_like
        ):
            route_label = "simple"
            routing_reason = "strict_trend_drift_pc1"
        else:
            route_label = "complex"
            if sentiment_like:
                routing_reason = "trend_drift_sentiment_blocked"
            elif seasonal_memory:
                routing_reason = "trend_drift_seasonal_memory"
            elif family_confidence != "high":
                routing_reason = "trend_drift_low_confidence"
            elif srr > 1.8:
                routing_reason = "trend_drift_slope_regime_change"
            else:
                routing_reason = "trend_drift_not_simple_enough"

    elif family == "structural_swing":
        route_label = "complex"
        if phase_count_raw > 1 or trend_type == "non_monotonic":
            routing_reason = "structural_swing_detected"
        elif pr >= 0.28 or sr >= 0.22 or reversal_evidence or prominent_internal_turn:
            routing_reason = "structural_swing_internal_evidence"
        elif srr > 1.8:
            routing_reason = "structural_swing_slope_regime_change"
        else:
            routing_reason = "structural_swing_family_block"

    elif family == "oscillatory_or_seasonal_regime":
        route_label = "complex"
        routing_reason = "oscillatory_or_seasonal_family"

    guidance = build_generation_guidance(route_label, family)

    return {
        "route_label": route_label,
        "is_simple": bool(route_label == "simple"),
        "routing_reason": routing_reason,
        "duration_months": int(duration_months),
        "trend_type": trend_type,
        "phase_count": int(phase_count_raw),
        "generation_guidance": guidance,
    }


# ----------------------------------------------------------------
# Main conversion function
# ----------------------------------------------------------------

def build_v2_anchor(
    image_key: str,
    v1_anchor: dict,
    csv_dir: Path,
) -> Optional[dict]:
    """
    Build a Stage 2 anchor from a Stage 1 anchor and cleaned CSV data.
    """
    csv_source = v1_anchor.get("csv_source", "")
    semantics  = VARIABLE_SEMANTICS.get(csv_source, DEFAULT_SEMANTICS)

    tw   = v1_anchor.get("time_window", {})
    vals = v1_anchor.get("values", {})
    chg  = v1_anchor.get("changes", {})
    ext  = v1_anchor.get("extremes", {})

    duration_months = tw.get("actual_duration_months", 0)
    pct_change      = chg.get("percentage_change") if chg.get("percentage_change_reason") == "valid" else None

    boundary_flags = compute_boundary_flags(v1_anchor)

    trend_type = "unknown"
    phases: List[dict] = []
    inflections: List[dict] = []
    morphology_signal = None
    binary_phase_signal = None

    csv_path = csv_dir / csv_source
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)

            if csv_source in ROLLING_SUM_FILES:
                df["trend_value"] = df["value"].rolling(window=12, min_periods=12).sum()
            else:
                df["trend_value"] = df["value"].rolling(window=12, min_periods=12).mean()

            start_dt = pd.Timestamp(tw.get("actual_start_date"))
            end_dt   = pd.Timestamp(tw.get("actual_end_date"))
            mask = (df["date"] >= start_dt) & (df["date"] <= end_dt)
            df_slice = df.loc[mask].copy()

            plot_col = v1_anchor.get("data_used", "value")
            if plot_col == "trend_value":
                df_for_phases = df_slice.dropna(subset=["trend_value"]).copy()
                if len(df_for_phases) < 3:
                    plot_col = "value"
                    df_for_phases = df_slice.dropna(subset=["value"]).copy()
            else:
                df_for_phases = df_slice.dropna(subset=["value"]).copy()

            raw_df = df_slice.dropna(subset=["value"]).copy()
            if len(df_for_phases) >= 2 and len(raw_df) >= 2:
                analysis_values = df_for_phases[plot_col].to_numpy(dtype=float)
                raw_values = raw_df["value"].to_numpy(dtype=float)
                morphology_base = classify_morphology_family(
                    csv_source=csv_source,
                    semantics=semantics,
                    analysis_values=analysis_values,
                    raw_values=raw_values,
                )
                morphology_signal = {
                    "family": morphology_base["family"],
                    "family_confidence": morphology_base["family_confidence"],
                    "analysis_col": plot_col,
                    "gate_metrics": morphology_base["gate_metrics"],
                    "modifier_flags": morphology_base["modifier_flags"],
                }

            if len(df_for_phases) >= 6:
                values_arr = df_for_phases[plot_col].to_numpy(dtype=float)
                dates_idx  = pd.DatetimeIndex(df_for_phases["date"])
                trend_type, phases, inflections = detect_trend_phases(
                    values_arr, dates_idx,
                    min_phase_months=max(3, duration_months // 8),
                )
            elif len(df_for_phases) >= 2:
                v_arr = df_for_phases[plot_col].to_numpy(dtype=float)
                direction = "increasing" if v_arr[-1] >= v_arr[0] else "decreasing"
                trend_type = f"monotonic_{direction}"
                phases = [{
                    "direction": direction,
                    "from_date": str(df_for_phases.iloc[0]["date"])[:10],
                    "to_date": str(df_for_phases.iloc[-1]["date"])[:10],
                    "start_value": round(float(v_arr[0]), 4),
                    "end_value": round(float(v_arr[-1]), 4),
                    "magnitude": round(float(v_arr[-1] - v_arr[0]), 4),
                }]
        except Exception:
            pass

    if trend_type == "unknown":
        start_v = vals.get("start_value", 0)
        end_v   = vals.get("end_value", 0)
        min_v   = vals.get("min_value", 0)
        max_v   = vals.get("max_value", 0)

        start_is_extreme = (start_v == min_v or start_v == max_v)
        end_is_extreme   = (end_v == min_v or end_v == max_v)

        if start_is_extreme and end_is_extreme:
            direction = "increasing" if end_v > start_v else "decreasing"
            trend_type = f"monotonic_{direction}"
        else:
            trend_type = "non_monotonic"

    if morphology_signal is None:
        fallback_values = np.array([
            to_float(vals.get("start_value")) if to_float(vals.get("start_value")) is not None else 0.0,
            to_float(vals.get("end_value")) if to_float(vals.get("end_value")) is not None else 0.0,
        ], dtype=float)
        morphology_base = classify_morphology_family(
            csv_source=csv_source,
            semantics=semantics,
            analysis_values=fallback_values,
            raw_values=fallback_values,
        )
        morphology_signal = {
            "family": morphology_base["family"],
            "family_confidence": "low",
            "analysis_col": v1_anchor.get("data_used", "value"),
            "gate_metrics": morphology_base["gate_metrics"],
            "modifier_flags": morphology_base["modifier_flags"],
        }

    binary_phase_signal = compute_binary_phase_signal(
        trend_type=trend_type,
        phases=phases,
        morphology_signal=morphology_signal,
    )

    complexity = compute_complexity_signal(
        trend_type=trend_type,
        duration_months=duration_months,
        morphology_signal=morphology_signal,
        binary_phase_signal=binary_phase_signal,
    )

    peak_info   = ext.get("peak", {})
    trough_info = ext.get("trough", {})

    v2 = {
        "variable_semantics": {
            "topic_category":    semantics["topic_category"],
            "unit_description":  semantics["unit_description"],
            "value_nature":      semantics["value_nature"],
            "higher_means":      semantics["higher_means"],
            "direction_note":    semantics["direction_note"],
            "closed_system_note": semantics.get("closed_system_note", ""),
        },

        "time_window": {
            "start": tw.get("actual_start_date", ""),
            "end":   tw.get("actual_end_date",   ""),
            "duration_months": duration_months,
        },

        "values": {
            "start_value": vals.get("start_value"),
            "end_value":   vals.get("end_value"),
        },

        "changes": {
            "absolute_change":   chg.get("absolute_change"),
            "percentage_change": round(pct_change, 4) if pct_change is not None else None,
        },

        "extremes": {
            "peak": {
                "date":  peak_info.get("date", ""),
                "value": peak_info.get("value"),
            },
            "trough": {
                "date":  trough_info.get("date", ""),
                "value": trough_info.get("value"),
            },
        },

        "boundary_flags": boundary_flags,

        "trend_structure": {
            "type":        trend_type,
            "phases":      phases,
            "inflections": inflections,
        },

        "morphology_signal": morphology_signal,
        "binary_phase_signal": binary_phase_signal,
        "complexity_signal": complexity,

        "is_high_risk_topic": csv_source in HIGH_RISK_TOPICS,
    }

    return v2


# ----------------------------------------------------------------
# CLI
# ----------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate Stage 2 morphology-aware anchors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--anchor_v1", default=str(DEFAULT_STAGE1_ANCHOR_FILE),
                   help="Path to the Stage 1 anchor JSON file.")
    p.add_argument("--semantic_pool", default=str(DEFAULT_SEMANTIC_POOL_FILE),
                   help="Path to semantic_pool.json; only these sample ids are processed.")
    p.add_argument("--csv_dir",   default=str(DEFAULT_CSV_DIR),
                   help="Directory containing cleaned CSV files.")
    p.add_argument("--output",    default=str(DEFAULT_STAGE2_ANCHOR_FILE),
                   help="Output path for the Stage 2 anchor JSON file.")
    p.add_argument("--exclude_high_risk", action="store_true",
                   help="Exclude samples whose CSV source belongs to HIGH_RISK_TOPICS.")
    p.add_argument("--dry_run",   action="store_true",
                   help="Print statistics without writing an output file.")
    p.add_argument("--limit",     type=int, default=None,
                   help="Process only the first N samples.")
    p.add_argument("--strict_align", action="store_true",
                   help="Check whether output keys exactly match semantic_pool input ids.")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 72)
    print("  Stage 2 anchor extraction (morphology-aware + binary PC)")
    print("  family gate -> fixed phase detector -> binary pc -> final simple/complex")
    print("=" * 72)

    v1_path = Path(args.anchor_v1)
    if not v1_path.exists():
        print(f"v1 anchor file does not exist: {v1_path.resolve()}")
        return
    with open(v1_path, "r", encoding="utf-8") as f:
        v1_anchors = json.load(f)
    print(f"Loaded v1 anchors: {len(v1_anchors)} entries")

    semantic_pool_path = Path(args.semantic_pool)
    if not semantic_pool_path.exists():
        print(f"semantic_pool file does not exist: {semantic_pool_path.resolve()}")
        return
    semantic_ids = load_semantic_pool_ids(str(semantic_pool_path))
    print(f"Loaded semantic_pool ids: {len(semantic_ids)} entries")

    csv_dir = Path(args.csv_dir)
    if not csv_dir.exists():
        print(f"CSV directory does not exist: {csv_dir.resolve()}; trend phase computation will be skipped.")

    keys = [k for k in semantic_ids if k in v1_anchors]
    missing_in_v1 = [k for k in semantic_ids if k not in v1_anchors]

    if missing_in_v1:
        print(f"{len(missing_in_v1)} semantic_pool ids are missing from v1 anchors")
        for x in missing_in_v1[:10]:
            print(f"  - {x}")

    if args.limit:
        keys = keys[:args.limit]

    v2_anchors = {}
    skipped = 0
    high_risk_n = 0
    non_mono_n = 0
    boundary_n = 0
    simple_n = 0
    complex_n = 0
    family_counts = {
        "trend_drift": 0,
        "structural_swing": 0,
        "oscillatory_or_seasonal_regime": 0,
    }

    for key in tqdm(keys, desc="Building v2 anchors"):
        v2 = build_v2_anchor(key, v1_anchors[key], csv_dir)
        if v2 is None:
            skipped += 1
            continue
        v2_anchors[key] = v2

        if v2["is_high_risk_topic"]:
            high_risk_n += 1
        if v2["trend_structure"]["type"] == "non_monotonic":
            non_mono_n += 1
        if v2["complexity_signal"]["route_label"] == "simple":
            simple_n += 1
        else:
            complex_n += 1
        family = v2.get("morphology_signal", {}).get("family")
        if family in family_counts:
            family_counts[family] += 1
        bf = v2["boundary_flags"]
        if bf["trough_at_window_start"] or bf["peak_at_window_start"]:
            boundary_n += 1

    if args.exclude_high_risk:
        v2_anchors = {k: v for k, v in v2_anchors.items() if not v.get("is_high_risk_topic", False)}

        high_risk_n = 0
        non_mono_n = 0
        boundary_n = 0
        simple_n = 0
        complex_n = 0
        family_counts = {
            "trend_drift": 0,
            "structural_swing": 0,
            "oscillatory_or_seasonal_regime": 0,
        }
        for v2 in v2_anchors.values():
            if v2["is_high_risk_topic"]:
                high_risk_n += 1
            if v2["trend_structure"]["type"] == "non_monotonic":
                non_mono_n += 1
            if v2["complexity_signal"]["route_label"] == "simple":
                simple_n += 1
            else:
                complex_n += 1
            family = v2.get("morphology_signal", {}).get("family")
            if family in family_counts:
                family_counts[family] += 1
            bf = v2["boundary_flags"]
            if bf["trough_at_window_start"] or bf["peak_at_window_start"]:
                boundary_n += 1

    print(f"\nSuccess: {len(v2_anchors)} entries / skipped: {skipped} entries")
    print(f"  High-risk topics: {high_risk_n} entries" + (" (excluded from output)" if args.exclude_high_risk else " (retained with flags)"))
    print(f"  Non-monotonic trends: {non_mono_n} entries")
    print(f"  final simple: {simple_n} entries")
    print(f"  final complex: {complex_n} entries")
    print(f"  family counts: trend={family_counts['trend_drift']}, swing={family_counts['structural_swing']}, oscillatory={family_counts['oscillatory_or_seasonal_regime']}")
    print(f"  Boundary extrema: {boundary_n} entries")

    if v2_anchors:
        sample_key = list(v2_anchors.keys())[0]
        sample_str = json.dumps(v2_anchors[sample_key], ensure_ascii=False)
        est_tokens = len(sample_str) // 4
        print(f"\nEstimated tokens per anchor: {est_tokens}")
        print(f"Original v1 anchor estimate: {len(json.dumps(v1_anchors[sample_key])) // 4} tokens")

    if args.strict_align:
        out_keys = set(v2_anchors.keys())
        in_keys = set(keys)
        missing_out = sorted(in_keys - out_keys)
        extra_out = sorted(out_keys - in_keys)

        print("\n[Alignment Check]")
        print(f"Input ids            : {len(in_keys)}")
        print(f"Output ids           : {len(out_keys)}")
        print(f"Missing output ids   : {len(missing_out)}")
        print(f"Extra output ids     : {len(extra_out)}")

        if missing_out:
            print("First 10 missing output ids:")
            for x in missing_out[:10]:
                print(f"  - {x}")

        if extra_out:
            print("First 10 extra output ids:")
            for x in extra_out[:10]:
                print(f"  - {x}")

    if not args.dry_run:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(v2_anchors, f, ensure_ascii=False, indent=2)
        print(f"\nSaved: {out_path.resolve()}")
    else:
        print("\n[DRY RUN] No file written. Example output (first item):")
        if v2_anchors:
            print(json.dumps(v2_anchors[list(v2_anchors.keys())[0]], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
