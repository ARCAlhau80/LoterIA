#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoterIA v3.0 - Demo das Melhorias Implementadas
Demonstra as funcionalidades avançadas baseadas no script original
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter
from pattern_analyzer import PatternAnalyzer

def demo_pattern_analysis():
    """Demonstração das análises de padrões"""
    print("🎲 LoterIA v3.0 - DEMO DAS MELHORIAS")
    print("=" * 60)
    print("Baseado nas técnicas do script original com:")
    print("✅ Análise de Divergência Posicional")
    print("✅ Padrões Posicionais Recorrentes") 
    print("✅ Diagnóstico de Filtros Condicionais")
    print("✅ Análise de Frequência Temporal")
    print("✅ Geração Inteligente de Combinações")
    print("=" * 60)
    
    try:
        # Inicializar analisador
        analyzer = PatternAnalyzer()
        
        print("\n🔍 1. ANÁLISE DE DIVERGÊNCIA POSICIONAL")
        print("-" * 50)
        print("Compara frequência histórica vs janela recente (últimos 15 concursos)")
        
        df_divergencia = analyzer.analisar_divergencia_posicional(janela_size=15)
        
        # Mostrar principais desvios
        print("\n🔥 NÚMEROS COM MAIOR DESVIO POSITIVO (mais frequentes que o normal):")
        positivos = df_divergencia[df_divergencia['status'].isin(['🔥 Muito acima', '⬆️ Acima'])]
        if not positivos.empty:
            top_positivos = positivos.nlargest(5, 'divergencia_pct')
            for _, row in top_positivos.iterrows():
                print(f"  Nº {row['numero']:02d} na {row['posicao']}: {row['divergencia_pct']:+.1f}% {row['status']}")
        else:
            print("  Nenhum desvio positivo significativo encontrado")
        
        print("\n❄️ NÚMEROS COM MAIOR DESVIO NEGATIVO (menos frequentes que o normal):")
        negativos = df_divergencia[df_divergencia['status'].isin(['❄️ Muito abaixo', '⬇️ Abaixo'])]
        if not negativos.empty:
            top_negativos = negativos.nsmallest(5, 'divergencia_pct')
            for _, row in top_negativos.iterrows():
                print(f"  Nº {row['numero']:02d} na {row['posicao']}: {row['divergencia_pct']:+.1f}% {row['status']}")
        else:
            print("  Nenhum desvio negativo significativo encontrado")
        
        print("\n🔍 2. ANÁLISE DE PADRÃO POSICIONAL RECORRENTE")
        print("-" * 50)
        print("Analisa padrões específicos e prevê quando podem reaparecer")
        
        # Exemplo: números 1 e 25 nas extremidades
        padrao_exemplo = {'numero_1': 1, 'numero_15': 25}
        print(f"Analisando padrão: {padrao_exemplo}")
        
        resultado_padrao = analyzer.analisar_padrao_posicional_recorrente(padrao_exemplo)
        
        if resultado_padrao:
            print(f"📊 Padrão encontrado em {resultado_padrao['total_ocorrencias']} concursos")
            print(f"📅 Última ocorrência: Concurso {resultado_padrao['ultima_ocorrencia']}")
            print(f"📈 Distância média entre ocorrências: {resultado_padrao['media_distancia']:.1f} concursos")
            print(f"🔮 Previsão próxima ocorrência: Concurso {resultado_padrao['concurso_previsto']} ± {resultado_padrao['margem_erro']}")
            
            if resultado_padrao['alerta_sazonal']:
                print("🚨 ALERTA SAZONAL: Padrão próximo de se repetir!")
            else:
                print("🟢 Fora da janela sazonal de reaparecimento")
        else:
            print("⚠️ Padrão não encontrado nos dados históricos")
        
        print("\n🔍 3. ANÁLISE DE FREQUÊNCIA TEMPORAL")
        print("-" * 50)
        print("Compara duas janelas temporais para identificar tendências")
        
        df_frequencia = analyzer.analisar_frequencia_temporal(frame_size=15)
        
        print("\n📈 NÚMEROS EM FORTE ALTA (aparecem mais recentemente):")
        altas = df_frequencia[df_frequencia['tendencia'].isin(['🔥 Forte alta', '⬆️ Alta'])]
        if not altas.empty:
            for _, row in altas.head(5).iterrows():
                print(f"  Nº {row['numero']:02d}: {row['diferenca']:+d} ocorrências {row['tendencia']}")
        else:
            print("  Nenhum número em alta significativa")
        
        print("\n📉 NÚMEROS EM FORTE BAIXA (aparecem menos recentemente):")
        baixas = df_frequencia[df_frequencia['tendencia'].isin(['❄️ Forte baixa', '⬇️ Baixa'])]
        if not baixas.empty:
            for _, row in baixas.head(5).iterrows():
                print(f"  Nº {row['numero']:02d}: {row['diferenca']:+d} ocorrências {row['tendencia']}")
        else:
            print("  Nenhum número em baixa significativa")
        
        print("\n🔍 4. DIAGNÓSTICO DE FILTROS CONDICIONAIS")
        print("-" * 50)
        print("Analisa estatísticas baseadas em filtros específicos")
        
        # Exemplo: analisar concursos com números pares específicos
        colunas_teste = ['soma_total']  # Usar apenas colunas que provavelmente existem
        
        # Simular análise de filtros (adaptado para nossa estrutura)
        print("Exemplo: Análise condicional simulada")
        print("(Esta análise seria mais rica com mais features no banco)")
        
        # Estatísticas básicas dos dados
        df_historico = analyzer.df_historico
        if 'concurso' in df_historico.columns:
            print(f"📊 Total de concursos analisados: {len(df_historico)}")
            print(f"📅 Período: Concurso {df_historico['concurso'].min()} a {df_historico['concurso'].max()}")
        
        print("\n🔍 5. GERAÇÃO DE COMBINAÇÕES INTELIGENTES")
        print("-" * 50)
        print("Gera combinações baseadas na análise de divergência posicional")
        
        df_combinacoes = analyzer.gerar_combinacoes_inteligentes(
            df_divergencia=df_divergencia,
            limite_combinacoes=3,
            usar_pesos=True
        )
        
        print(f"\n🎯 {len(df_combinacoes)} combinações geradas com base nos padrões:")
        for i, row in df_combinacoes.iterrows():
            numeros = [row[f'numero_{j}'] for j in range(1, 16)]
            numeros_str = ' - '.join(f'{n:02d}' for n in sorted(numeros))
            print(f"  Combinação {i+1}: {numeros_str}")
            print(f"    Método: {row['metodo']} | Gerada: {row['data_geracao']}")
        
        print("\n✅ RESUMO DAS MELHORIAS IMPLEMENTADAS:")
        print("-" * 50)
        print("🔥 Sistema de análise de divergência posicional")
        print("🎯 Detecção de padrões recorrentes com previsão temporal")
        print("📊 Diagnóstico avançado de filtros condicionais")
        print("📈 Análise comparativa de frequência temporal")
        print("🧠 Geração inteligente de combinações com pesos")
        print("💾 Persistência de análises no banco de dados")
        print("📝 Sistema de logging e rastreabilidade")
        
        print("\n🚀 PRÓXIMOS PASSOS SUGERIDOS:")
        print("-" * 50)
        print("1. Execute 'python main_v3.py' para usar o sistema completo")
        print("2. Treine o modelo neural com 'Opção 1' do menu")
        print("3. Gere predições híbridas com 'Opção 2' do menu")
        print("4. Explore análises específicas com 'Opções 3-5'")
        
    except Exception as e:
        print(f"❌ Erro durante demonstração: {e}")
        print("Verifique se o banco de dados está disponível em 'data/loteria.db'")

def main():
    """Função principal da demonstração"""
    try:
        demo_pattern_analysis()
    except KeyboardInterrupt:
        print("\n\n⏹️ Demonstração interrompida pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro fatal: {e}")

if __name__ == "__main__":
    main()
