#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para popular o banco SQLite com dados de teste
"""

import sqlite3
import random
import numpy as np
from datetime import datetime, timedelta

def generate_lotofacil_numbers():
    """Gera uma combinação válida da Lotofácil (15 números únicos de 1 a 25)"""
    return sorted(random.sample(range(1, 26), 15))

def calculate_basic_stats(numbers):
    """Calcula estatísticas básicas dos números"""
    primos = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    fibonacci = [1, 2, 3, 5, 8, 13, 21]
    
    qtde_primos = sum(1 for n in numbers if n in primos)
    qtde_fibonacci = sum(1 for n in numbers if n in fibonacci)
    qtde_impares = sum(1 for n in numbers if n % 2 == 1)
    soma_total = sum(numbers)
    
    # Quintis (1-5, 6-10, 11-15, 16-20, 21-25)
    quintil1 = sum(1 for n in numbers if 1 <= n <= 5)
    quintil2 = sum(1 for n in numbers if 6 <= n <= 10)
    quintil3 = sum(1 for n in numbers if 11 <= n <= 15)
    quintil4 = sum(1 for n in numbers if 16 <= n <= 20)
    quintil5 = sum(1 for n in numbers if 21 <= n <= 25)
    
    # Faixas
    faixa_baixa = sum(1 for n in numbers if 1 <= n <= 8)
    faixa_media = sum(1 for n in numbers if 9 <= n <= 17)
    faixa_alta = sum(1 for n in numbers if 18 <= n <= 25)
    
    # Gaps e sequências
    gaps = []
    for i in range(1, len(numbers)):
        gaps.append(numbers[i] - numbers[i-1])
    qtde_gaps = len([g for g in gaps if g == 1])
    
    # Sequências
    seq_count = 0
    current_seq = 1
    for i in range(1, len(numbers)):
        if numbers[i] == numbers[i-1] + 1:
            current_seq += 1
        else:
            if current_seq >= 2:
                seq_count += 1
            current_seq = 1
    if current_seq >= 2:
        seq_count += 1
    
    # Múltiplos de 3
    qtde_multiplos3 = sum(1 for n in numbers if n % 3 == 0)
    
    # Distância entre extremos
    distancia_extremos = numbers[-1] - numbers[0]
    
    return {
        'QtdePrimos': qtde_primos,
        'QtdeFibonacci': qtde_fibonacci,
        'QtdeImpares': qtde_impares,
        'SomaTotal': soma_total,
        'Quintil1': quintil1,
        'Quintil2': quintil2,
        'Quintil3': quintil3,
        'Quintil4': quintil4,
        'Quintil5': quintil5,
        'QtdeGaps': qtde_gaps,
        'QtdeRepetidos': 0,  # Para dados simulados
        'SEQ': seq_count,
        'DistanciaExtremos': distancia_extremos,
        'ParesSequencia': 0,  # Simplificado
        'QtdeMultiplos3': qtde_multiplos3,
        'ParesSaltados': 0,  # Simplificado
        'Faixa_Baixa': faixa_baixa,
        'Faixa_Media': faixa_media,
        'Faixa_Alta': faixa_alta,
        'RepetidosMesmaPosicao': 0  # Para dados simulados
    }

def create_database():
    """Cria e popula o banco SQLite com dados de teste"""
    print("🗃️ Criando banco de dados SQLite com dados de teste...")
    
    conn = sqlite3.connect('data/loteria.db')
    cursor = conn.cursor()
    
    # Criar tabela
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS resultados (
        Concurso INTEGER PRIMARY KEY,
        DataConcurso TEXT,
        N1 INTEGER, N2 INTEGER, N3 INTEGER, N4 INTEGER, N5 INTEGER,
        N6 INTEGER, N7 INTEGER, N8 INTEGER, N9 INTEGER, N10 INTEGER,
        N11 INTEGER, N12 INTEGER, N13 INTEGER, N14 INTEGER, N15 INTEGER,
        QtdePrimos INTEGER,
        QtdeFibonacci INTEGER,
        QtdeImpares INTEGER,
        SomaTotal INTEGER,
        Quintil1 INTEGER,
        Quintil2 INTEGER,
        Quintil3 INTEGER,
        Quintil4 INTEGER,
        Quintil5 INTEGER,
        QtdeGaps INTEGER,
        QtdeRepetidos INTEGER,
        SEQ INTEGER,
        DistanciaExtremos INTEGER,
        ParesSequencia INTEGER,
        QtdeMultiplos3 INTEGER,
        ParesSaltados INTEGER,
        Faixa_Baixa INTEGER,
        Faixa_Media INTEGER,
        Faixa_Alta INTEGER,
        RepetidosMesmaPosicao INTEGER
    )
    ''')
    
    # Gerar dados de teste
    start_date = datetime(2020, 1, 1)
    
    print("📊 Gerando 2000 registros de teste...")
    
    for i in range(2000):
        concurso = 1000 + i
        data_concurso = (start_date + timedelta(days=i*2)).strftime('%Y-%m-%d')
        
        # Gerar números da lotofácil
        numbers = generate_lotofacil_numbers()
        stats = calculate_basic_stats(numbers)
        
        # Inserir no banco
        cursor.execute('''
        INSERT INTO resultados VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        ''', (
            concurso, data_concurso,
            *numbers,  # N1 a N15
            stats['QtdePrimos'],
            stats['QtdeFibonacci'],
            stats['QtdeImpares'],
            stats['SomaTotal'],
            stats['Quintil1'],
            stats['Quintil2'],
            stats['Quintil3'],
            stats['Quintil4'],
            stats['Quintil5'],
            stats['QtdeGaps'],
            stats['QtdeRepetidos'],
            stats['SEQ'],
            stats['DistanciaExtremos'],
            stats['ParesSequencia'],
            stats['QtdeMultiplos3'],
            stats['ParesSaltados'],
            stats['Faixa_Baixa'],
            stats['Faixa_Media'],
            stats['Faixa_Alta'],
            stats['RepetidosMesmaPosicao']
        ))
        
        if (i + 1) % 500 == 0:
            print(f"   ✅ {i + 1} registros inseridos...")
    
    conn.commit()
    conn.close()
    
    print("✅ Banco de dados criado com sucesso!")
    print(f"📈 Total: 2000 registros inseridos")
    print(f"🎲 Concursos: 1000 até 2999")
    print(f"📅 Período: 2020-01-01 até {(start_date + timedelta(days=1999*2)).strftime('%Y-%m-%d')}")

if __name__ == "__main__":
    create_database()
