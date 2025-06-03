#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoterIA v3.0 - Sistema AVANÇADO de Predição de Loteria com Análise de Padrões
Versão 3.0.0 - Integração com Pattern Analyzer e Técnicas Avançadas
Desenvolvido com TensorFlow + Análise de Padrões Inteligente
"""

import os
import sys
import numpy as np
import pandas as pd
import sqlite3
import pyodbc
import logging
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from collections import Counter

# Configurar TensorFlow para compatibilidade
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Importar nosso analisador de padrões
from pattern_analyzer import PatternAnalyzer

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class LoterIAConfig:
    """Configurações globais do sistema"""
    DB_TYPE: str = "sqlserver"  # sqlserver ou sqlite - CORRIGIDO: usando SQL Server
    SQL_SERVER: str = "DESKTOP-K6JPBDS"
    SQL_DATABASE: str = "LOTOFACIL"
    SQL_DRIVER: str = "ODBC Driver 17 for SQL Server"
    SQLITE_PATH: str = "data/loteria.db"
    
    # Configurações do modelo
    MODEL_PATH: str = "models/loteria_model_v3.h5"
    RESULTS_PATH: str = "results/"
    
    # Parâmetros de treinamento
    EPOCHS: int = 100
    BATCH_SIZE: int = 32
    VALIDATION_SPLIT: float = 0.2

class DatabaseManager:
    """Gerenciador de conexão com banco de dados"""
    
    def __init__(self, config: LoterIAConfig):
        self.config = config
        self.connection = None
    
    def connect(self):
        """Estabelece conexão com o banco"""
        try:
            if self.config.DB_TYPE.lower() == "sqlserver":
                connection_string = (
                    f"DRIVER={{{self.config.SQL_DRIVER}}};"
                    f"SERVER={self.config.SQL_SERVER};"
                    f"DATABASE={self.config.SQL_DATABASE};"
                    f"Trusted_Connection=yes;"
                )
                self.connection = pyodbc.connect(connection_string)
            else:
                self.connection = sqlite3.connect(self.config.SQLITE_PATH)
            
            logger.info("✅ Conexão com banco estabelecida")
            return self.connection
            
        except Exception as e:
            logger.error(f"❌ Erro de conexão: {e}")
            return None
    
    def close(self):
        """Fecha conexão com banco"""
        if self.connection:
            self.connection.close()
            logger.info("🔌 Conexão com banco fechada")

class DataProcessor:
    """Processador de dados com recursos avançados"""
    
    def __init__(self, config: LoterIAConfig):
        self.config = config
        self.db_manager = DatabaseManager(config)
        self.scaler = StandardScaler()
    
    def load_data(self) -> pd.DataFrame:
        """Carrega dados históricos"""
        try:
            conn = self.db_manager.connect()
            if self.db_manager.config.DB_TYPE.lower() == "sqlserver":
                # Query para SQL Server com tabela Resultados_INT
                query = """
                SELECT 
                    Concurso, data_sorteio,
                    N1, N2, N3, N4, N5, N6, N7, N8, N9, N10, N11, N12, N13, N14, N15,
                    QtdePrimos, QtdeFibonacci, QtdeImpares, SomaTotal,
                    Quintil1, Quintil2, Quintil3, Quintil4, Quintil5,
                    QtdeGaps, QtdeRepetidos, SEQ, DistanciaExtremos,
                    ParesSequencia, QtdeMultiplos3, ParesSaltados,
                    Faixa_Baixa, Faixa_Media, Faixa_Alta, RepetidosMesmaPosicao
                FROM Resultados_INT 
                ORDER BY Concurso ASC
                """
            else:
                # Query para SQLite (fallback)
                query = """
                SELECT * FROM resultados 
                ORDER BY concurso ASC
                """
            
            df = pd.read_sql_query(query, conn)
            logger.info(f"📊 Dados carregados: {len(df)} registros")
            
            # Normalizar nomes das colunas
            if self.db_manager.config.DB_TYPE.lower() == "sqlserver":
                df = df.rename(columns={'Concurso': 'concurso'})
                # Converter N1-N15 para n1-n15 para compatibilidade com PatternAnalyzer
                rename_dict = {f'N{i}': f'n{i}' for i in range(1, 16)}
                df = df.rename(columns=rename_dict)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar dados: {e}")
            raise
        finally:
            self.db_manager.close()
    
    def extract_features_enhanced(self, df: pd.DataFrame) -> np.ndarray:
        """Extrai features avançadas incluindo análise de padrões"""
        logger.info("🔧 Extraindo features avançadas...")
        
        features = []
        
        for _, row in df.iterrows():
            # Acessar colunas n1-n15 (minúsculas - após normalização)
            numeros = [row[f'n{i}'] for i in range(1, 16)]
            numeros = sorted(numeros)
            
            # Features básicas
            feature_row = []
            
            # 1. Números individuais (15 features)
            feature_row.extend(numeros)
            
            # 2. Features estatísticas básicas
            feature_row.append(sum(numeros))  # Soma total
            feature_row.append(np.mean(numeros))  # Média
            feature_row.append(np.std(numeros))  # Desvio padrão
            feature_row.append(max(numeros) - min(numeros))  # Range
            
            # 3. Features de padrão
            pares = sum(1 for n in numeros if n % 2 == 0)
            impares = 15 - pares
            feature_row.extend([pares, impares])
            
            # 4. Features de distribuição por dezenas
            dezena1 = sum(1 for n in numeros if 1 <= n <= 5)
            dezena2 = sum(1 for n in numeros if 6 <= n <= 10)
            dezena3 = sum(1 for n in numeros if 11 <= n <= 15)
            dezena4 = sum(1 for n in numeros if 16 <= n <= 20)
            dezena5 = sum(1 for n in numeros if 21 <= n <= 25)
            feature_row.extend([dezena1, dezena2, dezena3, dezena4, dezena5])
            
            # 5. Features de sequência
            gaps = [numeros[i+1] - numeros[i] for i in range(14)]
            feature_row.append(np.mean(gaps))  # Gap médio
            feature_row.append(max(gaps))  # Gap máximo
            feature_row.append(sum(1 for g in gaps if g == 1))  # Sequências consecutivas
            
            # 6. Features de números primos
            primos = [2, 3, 5, 7, 11, 13, 17, 19, 23]
            qtd_primos = sum(1 for n in numeros if n in primos)
            feature_row.append(qtd_primos)
            
            # 7. Features de Fibonacci
            fibonacci = [1, 2, 3, 5, 8, 13, 21]
            qtd_fibonacci = sum(1 for n in numeros if n in fibonacci)
            feature_row.append(qtd_fibonacci)
            
            # 8. Features temporais (posição relativa no histórico)
            feature_row.append(row['concurso'] / df['concurso'].max())
            
            features.append(feature_row)
        
        features_array = np.array(features, dtype=np.float32)
        
        # Normalizar features
        features_normalized = self.scaler.fit_transform(features_array)
        
        logger.info(f"✅ Features extraídas: shape {features_normalized.shape}")
        return features_normalized
    
    def create_sequences(self, features: np.ndarray, sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Cria sequências temporais para treinamento"""
        logger.info(f"🔄 Criando sequências temporais (length={sequence_length})")
        
        X, y = [], []
        
        for i in range(sequence_length, len(features)):
            X.append(features[i-sequence_length:i])
            y.append(features[i][:15])  # Apenas os 15 números
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        logger.info(f"✅ Sequências criadas: X{X.shape}, y{y.shape}")
        return X, y

class LoterIAModelV3:
    """Modelo neural avançado para LoterIA v3.0"""
    
    def __init__(self, config: LoterIAConfig):
        self.config = config
        self.model = None
        
    def build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Constrói modelo neural avançado"""
        logger.info(f"🏗️ Construindo modelo neural v3.0 com input_shape: {input_shape}")
        
        model = keras.Sequential([
            # Camada de entrada LSTM para sequências temporais
            keras.layers.LSTM(128, return_sequences=True, input_shape=input_shape),
            keras.layers.Dropout(0.2),
            keras.layers.BatchNormalization(),
            
            # Segunda camada LSTM
            keras.layers.LSTM(64, return_sequences=False),
            keras.layers.Dropout(0.2),
            keras.layers.BatchNormalization(),
            
            # Camadas densas para processamento final
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.BatchNormalization(),
            
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.1),
            
            # Camada de saída para 15 números (0-1 normalizado)
            keras.layers.Dense(15, activation='sigmoid')
        ])
          # Compilar modelo
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )
        
        self.model = model
        logger.info("✅ Modelo construído com sucesso")
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Treina o modelo"""
        logger.info("🚀 Iniciando treinamento do modelo neural...")
        
        # Callbacks para melhor treinamento
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.00001),
            keras.callbacks.ModelCheckpoint(self.config.MODEL_PATH, save_best_only=True)
        ]
        
        # Treinar modelo
        history = self.model.fit(
            X_train, y_train,
            epochs=self.config.EPOCHS,
            batch_size=self.config.BATCH_SIZE,
            validation_split=self.config.VALIDATION_SPLIT,
            callbacks=callbacks,
            verbose=1
        )
        
        # Salvar modelo
        self.model.save(self.config.MODEL_PATH)
        logger.info(f"💾 Modelo salvo em: {self.config.MODEL_PATH}")
        
        return history.history
    
    def load_model(self) -> bool:
        """Carrega modelo treinado"""
        try:
            if os.path.exists(self.config.MODEL_PATH):
                self.model = keras.models.load_model(self.config.MODEL_PATH)
                logger.info(f"✅ Modelo carregado: {self.config.MODEL_PATH}")
                return True
            else:
                logger.warning(f"⚠️ Arquivo de modelo não encontrado: {self.config.MODEL_PATH}")
                return False
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo: {e}")
            return False

class LoterIAPredictorV3:
    """Sistema de predição avançado v3.0 com análise de padrões"""
    
    def __init__(self, config: LoterIAConfig):
        self.config = config
        self.data_processor = DataProcessor(config)
        self.model = LoterIAModelV3(config)
        self.pattern_analyzer = PatternAnalyzer(use_sqlserver_data=True, db_manager=DatabaseManager(config))
    
    def train_complete_system(self) -> Dict:
        """Treina o sistema completo"""
        logger.info("🎯 Iniciando processo de treinamento completo...")
        
        # 1. Carregar dados
        df = self.data_processor.load_data()
        
        # 2. Extrair features
        features = self.data_processor.extract_features_enhanced(df)
        
        # 3. Criar sequências temporais
        X, y = self.data_processor.create_sequences(features)
        
        # 4. Construir modelo
        model = self.model.build_model(input_shape=(X.shape[1], X.shape[2]))
        
        # 5. Treinar modelo
        history = self.model.train(X, y)
        
        logger.info("🎉 Treinamento completo finalizado!")
        return history
    
    def predict_next_draw(self, num_predictions: int = 5) -> List[Dict]:
        """Gera predições inteligentes combinando IA + análise de padrões"""
        logger.info(f"🔮 Gerando {num_predictions} predições inteligentes...")
        
        # 1. Carregar dados atuais
        df = self.data_processor.load_data()
        
        # 2. Executar análises de padrões
        logger.info("🔍 Executando análise de padrões...")
        df_divergencia = self.pattern_analyzer.analisar_divergencia_posicional(janela_size=15)
        df_frequencia = self.pattern_analyzer.analisar_frequencia_temporal(frame_size=15)
        
        # 3. Gerar predições da IA
        logger.info("🤖 Gerando predições com IA...")
        ai_predictions = self._generate_ai_predictions(df, num_predictions)
        
        # 4. Gerar combinações baseadas em padrões
        logger.info("🎯 Gerando combinações baseadas em padrões...")
        df_combinacoes_padroes = self.pattern_analyzer.gerar_combinacoes_inteligentes(
            df_divergencia=df_divergencia,
            limite_combinacoes=num_predictions,
            usar_pesos=True
        )
        
        # 5. Combinar resultados
        predicoes = []
        
        for i in range(num_predictions):
            predicao = {
                'id': i + 1,
                'numeros': [],
                'metodo': 'Híbrido',
                'confianca': 0.5,
                'timestamp': datetime.now()
            }
            
            # Priorizar predições da IA se disponíveis
            if i < len(ai_predictions):
                predicao['numeros'] = ai_predictions[i]
                predicao['metodo'] = 'IA Neural'
                predicao['confianca'] = 0.7
            
            # Usar análise de padrões como fallback ou complemento
            elif i < len(df_combinacoes_padroes):
                pattern_nums = df_combinacoes_padroes.iloc[i][[f'n{j}' for j in range(1, 16)]].values
                predicao['numeros'] = sorted(pattern_nums)
                predicao['metodo'] = 'Análise de Padrões'
                predicao['confianca'] = 0.6
            
            # Garantir 15 números únicos
            while len(predicao['numeros']) < 15:
                num_aleatorio = np.random.randint(1, 26)
                if num_aleatorio not in predicao['numeros']:
                    predicao['numeros'].append(num_aleatorio)
            
            predicao['numeros'] = sorted(predicao['numeros'][:15])
            
            # Adicionar métricas de análise
            self._add_analysis_metrics(predicao, df_frequencia, df_divergencia)
            
            predicoes.append(predicao)
        
        return predicoes
    
    def _generate_ai_predictions(self, df: pd.DataFrame, num_predictions: int) -> List[List[int]]:
        """Gera predições usando IA neural"""
        if not self.model.load_model():
            logger.warning("⚠️ Modelo não carregado, usando predições aleatórias")
            return []
        
        try:            # Recompilar modelo com configurações atuais
            self.model.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='mean_squared_error',
                metrics=['mean_absolute_error']
            )
            logger.info("🔄 Modelo recompilado com configurações atuais")
            
            # Preparar dados para predição
            features = self.data_processor.extract_features_enhanced(df)
            
            # Usar últimas sequências para predição
            sequence_length = 10
            if len(features) >= sequence_length:
                last_sequence = features[-sequence_length:].reshape(1, sequence_length, -1)
                
                predictions = []
                for _ in range(num_predictions):
                    # Predição neural
                    pred = self.model.model.predict(last_sequence, verbose=0)[0]
                    
                    # Converter predição para números de 1-25
                    pred_numbers = (pred * 25 + 1).astype(int)
                    pred_numbers = np.clip(pred_numbers, 1, 25)
                    
                    # Garantir 15 números únicos
                    unique_numbers = []
                    for num in pred_numbers:
                        if num not in unique_numbers and len(unique_numbers) < 15:
                            unique_numbers.append(int(num))
                    
                    # Completar se necessário
                    while len(unique_numbers) < 15:
                        num = np.random.randint(1, 26)
                        if num not in unique_numbers:
                            unique_numbers.append(num)
                    
                    predictions.append(sorted(unique_numbers[:15]))
                    
                    # Adicionar ruído para próxima predição
                    last_sequence = last_sequence + np.random.normal(0, 0.01, last_sequence.shape)
                
                logger.info("✅ Predições IA geradas com sucesso")
                return predictions
            
        except Exception as e:
            logger.error(f"❌ Erro na predição IA: {e}")
        
        return []
    
    def _add_analysis_metrics(self, predicao: Dict, 
                            df_frequencia: pd.DataFrame, 
                            df_divergencia: pd.DataFrame) -> None:
        """Adiciona métricas de análise à predição"""
        numeros = predicao['numeros']
        
        # Análise de tendência
        tendencias = []
        for num in numeros:
            freq_info = df_frequencia[df_frequencia['numero'] == num]
            if not freq_info.empty:
                tendencias.append(freq_info.iloc[0]['tendencia'])
        
        # Análise de divergência
        divergencias = []
        for num in numeros:
            div_info = df_divergencia[df_divergencia['numero'] == num]
            if not div_info.empty:
                divergencias.append(div_info.iloc[0]['status'])
        
        # Estatísticas
        predicao['estatisticas'] = {
            'soma_total': sum(numeros),
            'media': np.mean(numeros),
            'pares': sum(1 for n in numeros if n % 2 == 0),
            'impares': sum(1 for n in numeros if n % 2 == 1),
            'tendencias': Counter(tendencias).most_common(),
            'status_divergencia': Counter(divergencias).most_common()
        }
    
    def save_predictions(self, predicoes: List[Dict], filename: str = None) -> str:
        """Salva predições em arquivo"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"predicao_v3_{timestamp}.txt"
        
        filepath = os.path.join(self.config.RESULTS_PATH, filename)
        
        # Criar diretório se não existir
        os.makedirs(self.config.RESULTS_PATH, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("🎲 LoterIA v3.0 - Predições Inteligentes\n")
            f.write("=" * 50 + "\n")
            f.write(f"Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
            f.write(f"Total de predições: {len(predicoes)}\n\n")
            
            for i, pred in enumerate(predicoes, 1):
                f.write(f"PREDIÇÃO {i} ({pred['metodo']}) - Confiança: {pred['confianca']:.1%}\n")
                f.write("-" * 30 + "\n")
                
                # Números formatados
                numeros_str = " - ".join(f"{n:02d}" for n in pred['numeros'])
                f.write(f"Números: {numeros_str}\n")
                
                # Estatísticas
                stats = pred.get('estatisticas', {})
                f.write(f"Soma: {stats.get('soma_total', 0)}\n")
                f.write(f"Pares/Ímpares: {stats.get('pares', 0)}/{stats.get('impares', 0)}\n")
                  # Tendências principais
                tendencias = stats.get('tendencias', [])
                if tendencias:
                    f.write(f"Tendências: {', '.join([f'{t[0]} ({t[1]})' for t in tendencias[:3]])}\n")
                
                f.write("\n")
        
        logger.info(f"💾 Predições salvas em: {filepath}")
        return filepath

def exibir_banner():
    """Exibe banner do sistema"""
    print("🎲 LoterIA v3.0 - Sistema Avançado de Predição de Loteria")
    print("=" * 60)

def exibir_menu():
    """Exibe menu principal"""
    print("\n📋 MENU PRINCIPAL:")
    print("1. 🚀 Treinar Modelo")
    print("2. 🔮 Gerar Predições")
    print("3. 🔍 Análise de Padrões")
    print("4. 📊 Análise de Divergência")
    print("5. 📈 Análise de Frequência Temporal")
    print("0. ❌ Sair")

def main():
    """Função principal"""
    config = LoterIAConfig()
    predictor = LoterIAPredictorV3(config)
    
    exibir_banner()
    
    while True:
        try:
            exibir_menu()
            opcao = input("👉 Escolha uma opção: ").strip()
            
            if opcao == "0":
                print("👋 Encerrando LoterIA v3.0...")
                break
                
            elif opcao == "1":
                print("🚀 Iniciando treinamento do modelo...")
                try:
                    history = predictor.train_complete_system()
                    print("✅ Treinamento concluído com sucesso!")
                except Exception as e:
                    logger.error(f"❌ Erro no treinamento: {e}")
                    print(f"❌ Erro: {e}")
                
            elif opcao == "2":
                try:
                    num_pred = int(input("Quantas predições gerar? (1-10): "))
                    if 1 <= num_pred <= 10:
                        predicoes = predictor.predict_next_draw(num_pred)
                        
                        # Exibir predições
                        print(f"\n🎯 {len(predicoes)} PREDIÇÕES GERADAS:")
                        print("=" * 50)
                        
                        for pred in predicoes:
                            numeros = " - ".join(f"{n:02d}" for n in pred['numeros'])
                            print(f"🎲 {pred['metodo']} (Confiança: {pred['confianca']:.1%}): {numeros}")
                        
                        # Salvar em arquivo
                        arquivo = predictor.save_predictions(predicoes)
                        print(f"💾 Resultados salvos em: {arquivo}")
                        
                    else:
                        print("⚠️ Número deve estar entre 1 e 10")
                        
                except ValueError:
                    print("⚠️ Entrada inválida")
                except Exception as e:
                    logger.error(f"❌ Erro fatal: {e}")
                    print(f"❌ Erro: {e}")
            
            elif opcao == "3":
                print("🔍 Executando análise de padrões...")
                try:
                    # Análise de divergência
                    df_div = predictor.pattern_analyzer.analisar_divergencia_posicional(15)
                    print(f"\n📊 Análise de divergência: {len(df_div)} registros")
                    
                    # Mostrar principais desvios positivos
                    print("\n🔥 TOP 5 DESVIOS POSITIVOS:")
                    positivos = df_div[df_div['divergencia_pct'] > 0].nlargest(5, 'divergencia_pct')
                    for _, row in positivos.iterrows():
                        print(f"  Nº {row['numero']:02d} na {row['posicao']}: +{row['divergencia_pct']:.1f}% {row['status']}")
                    
                    # Mostrar principais desvios negativos
                    print("\n❄️ TOP 5 DESVIOS NEGATIVOS:")
                    negativos = df_div[df_div['divergencia_pct'] < 0].nsmallest(5, 'divergencia_pct')
                    for _, row in negativos.iterrows():
                        print(f"  Nº {row['numero']:02d} na {row['posicao']}: {row['divergencia_pct']:.1f}% {row['status']}")
                        
                except Exception as e:
                    logger.error(f"❌ Erro na análise: {e}")
                    print(f"❌ Erro: {e}")
            
            elif opcao == "4":
                print("📊 Executando análise de divergência detalhada...")
                try:
                    janela = int(input("Tamanho da janela (padrão 15): ") or 15)
                    df_div = predictor.pattern_analyzer.analisar_divergencia_posicional(janela)
                    
                    # Mostrar principais desvios
                    print("\n🔥 MAIORES DESVIOS POSITIVOS:")
                    positivos = df_div[df_div['status'].isin(['🔥 Muito acima', '⬆️ Acima'])]
                    print(positivos.nlargest(10, 'divergencia_pct')[
                        ['numero', 'posicao', 'divergencia_pct', 'status']
                    ].to_string(index=False))
                    
                    print("\n❄️ MAIORES DESVIOS NEGATIVOS:")
                    negativos = df_div[df_div['status'].isin(['❄️ Muito abaixo', '⬇️ Abaixo'])]
                    print(negativos.nsmallest(10, 'divergencia_pct')[
                        ['numero', 'posicao', 'divergencia_pct', 'status']
                    ].to_string(index=False))
                    
                except ValueError:
                    print("⚠️ Valor inválido para janela")
                except Exception as e:
                    logger.error(f"❌ Erro na análise: {e}")
                    print(f"❌ Erro: {e}")
            
            elif opcao == "5":
                print("📈 Executando análise de frequência temporal...")
                try:
                    janela = int(input("Tamanho das janelas (padrão 15): ") or 15)
                    df_freq = predictor.pattern_analyzer.analisar_frequencia_temporal(janela)
                    
                    print("\n📈 NÚMEROS EM ALTA:")
                    altas = df_freq[df_freq['tendencia'].isin(['🔥 Forte alta', '⬆️ Alta'])]
                    if not altas.empty:
                        print(altas[['numero', 'diferenca', 'tendencia']].to_string(index=False))
                    else:
                        print("Nenhum número em alta significativa")
                    
                    print("\n📉 NÚMEROS EM BAIXA:")
                    baixas = df_freq[df_freq['tendencia'].isin(['❄️ Forte baixa', '⬇️ Baixa'])]
                    if not baixas.empty:
                        print(baixas[['numero', 'diferenca', 'tendencia']].to_string(index=False))
                    else:
                        print("Nenhum número em baixa significativa")
                        
                except ValueError:
                    print("⚠️ Valor inválido para janela")
                except Exception as e:
                    logger.error(f"❌ Erro na análise: {e}")
                    print(f"❌ Erro: {e}")
            
            else:
                print("⚠️ Opção inválida")
                
        except KeyboardInterrupt:
            print("\n👋 Operação cancelada pelo usuário")
            break
        except Exception as e:
            logger.error(f"❌ Erro inesperado: {e}")
            print(f"❌ Erro inesperado: {e}")

if __name__ == "__main__":
    main()
