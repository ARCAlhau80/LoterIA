#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoterIA - Sistema de Predição de Loteria com Inteligência Artificial
==================================================================

Módulo principal do sistema LoterIA para predição de resultados de loteria
usando técnicas de Machine Learning e Deep Learning.

Autor: Sistema LoterIA
Data: 2025-05-31
"""

# Importações principais
from pickle import NONE
import pandas as pd
import numpy as np
import pyodbc
import sqlite3
import time
import itertools
from sqlalchemy import create_engine
from tqdm import tqdm
from datetime import datetime
from itertools import combinations
from sklearn.preprocessing import StandardScaler
import os
import sys

# Configurar TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduzir logs do TensorFlow
import tensorflow as tf

# Configuração global
class LoterIAConfig:
    """Configurações globais do sistema LoterIA"""
    
    # Configurações do banco de dados
    DB_TYPE = "sqlserver"  # sqlite ou sqlserver
    SQLITE_DB_PATH = "data/loteria.db"
    
    # Configurações SQL Server (dados reais)
    SQL_SERVER = "DESKTOP-K6JPBDS"
    SQL_DATABASE = "Lotofacil"
    SQL_DRIVER = "ODBC Driver 17 for SQL Server"
    
    # Configurações do modelo
    MODEL_SAVE_PATH = "models/"
    DATA_PATH = "data/"
    RESULTS_PATH = "results/"
    
    # Configurações de predição
    DEFAULT_NUMBERS_TO_PREDICT = 15  # Para LOTOFACIL
    DEFAULT_RANGE = (1, 25)  # Range de números da LOTOFACIL
    
    @classmethod
    def ensure_directories(cls):
        """Cria diretórios necessários se não existirem"""
        directories = [cls.MODEL_SAVE_PATH, cls.DATA_PATH, cls.RESULTS_PATH]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"📁 Diretório criado/verificado: {directory}")

class DatabaseManager:
    """Gerenciador de conexões com banco de dados"""
    
    def __init__(self, config: LoterIAConfig):
        self.config = config
        self.connection = None
        
    def connect(self):
        """Estabelece conexão com o banco de dados"""
        try:
            if self.config.DB_TYPE.lower() == "sqlite":
                self.connection = sqlite3.connect(self.config.SQLITE_DB_PATH)
                print("✅ Conectado ao SQLite")
            elif self.config.DB_TYPE.lower() == "sqlserver":
                conn_str = f"DRIVER={{{self.config.SQL_DRIVER}}};SERVER={self.config.SQL_SERVER};DATABASE={self.config.SQL_DATABASE};Trusted_Connection=yes;"
                self.connection = pyodbc.connect(conn_str)
                print("✅ Conectado ao SQL Server")
            else:
                raise ValueError(f"Tipo de banco não suportado: {self.config.DB_TYPE}")
                
        except Exception as e:
            print(f"❌ Erro ao conectar ao banco: {e}")
            raise
            
    def disconnect(self):
        """Fecha conexão com o banco"""
        if self.connection:
            self.connection.close()
            print("🔌 Conexão fechada")
            
    def execute_query(self, query: str, params=None):
        """Executa uma query e retorna os resultados"""
        try:
            if not self.connection:
                self.connect()
                
            if params:
                cursor = self.connection.execute(query, params)
            else:
                cursor = self.connection.execute(query)
                
            return cursor.fetchall()
            
        except Exception as e:
            print(f"❌ Erro ao executar query: {e}")
            raise

class DataProcessor:
    """Processador de dados para análise e preparação"""
      def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        
    def load_historical_data(self) -> pd.DataFrame:
        """Carrega dados históricos de resultados da tabela Resultados_INT"""
        print("📊 Carregando dados históricos do SQL Server...")
        
        try:
            # Query para carregar dados da tabela real com TODAS as features disponíveis
            query = """
            SELECT TOP 2000
                Concurso,
                Data_Sorteio as DataSorteio,
                N1, N2, N3, N4, N5, N6, N7, N8, N9, N10,
                N11, N12, N13, N14, N15,
                QtdePrimos,
                QtdeFibonacci,
                QtdeImpares,
                SomaTotal,
                Quintil1,
                Quintil2,
                Quintil3,
                Quintil4,
                Quintil5,
                QtdeGaps,
                QtdeRepetidos,
                SEQ,
                DistanciaExtremos,
                ParesSequencia,
                QtdeMultiplos3,
                ParesSaltados,
                Faixa_Baixa,
                Faixa_Media,
                Faixa_Alta,
                RepetidosMesmaPosicao
            FROM Resultados_INT
            ORDER BY Concurso DESC
            """
            
            # Conectar e executar query
            self.db_manager.connect()
            
            # Usar pandas para carregar diretamente do SQL Server
            if self.db_manager.config.DB_TYPE.lower() == "sqlserver":
                import urllib.parse
                
                # Criar connection string para SQLAlchemy
                params = urllib.parse.quote_plus(
                    f"DRIVER={{{self.db_manager.config.SQL_DRIVER}}};"
                    f"SERVER={self.db_manager.config.SQL_SERVER};"
                    f"DATABASE={self.db_manager.config.SQL_DATABASE};"
                    f"Trusted_Connection=yes;"
                )
                engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")
                
                # Carregar dados com pandas
                df = pd.read_sql(query, engine)
                
                # Converter DataSorteio para datetime
                df['DataSorteio'] = pd.to_datetime(df['DataSorteio'])
                
                print(f"✅ {len(df)} registros carregados da tabela Resultados_INT")
                print(f"📅 Período: {df['DataSorteio'].min()} até {df['DataSorteio'].max()}")
                print(f"🎲 Concursos: {df['Concurso'].min()} até {df['Concurso'].max()}")
                
                return df
            else:                # Fallback para SQLite
                results = self.db_manager.execute_query(query)
                
                if not results:
                    print("⚠️ Nenhum dado histórico encontrado")
                    return self._create_sample_data()
                
                # Criar DataFrame com as colunas corretas
                columns = ['Concurso', 'DataSorteio'] + [f'Num{i}' for i in range(1, 16)]
                df = pd.DataFrame(results, columns=columns[:len(results[0])])
                
                print(f"✅ {len(df)} registros carregados")
                return df
            
        except Exception as e:
            print(f"❌ Erro ao carregar dados do SQL Server: {e}")
            print("🔄 Tentando usar dados de exemplo...")
            return self._create_sample_data()
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Cria dados de exemplo para desenvolvimento"""
        print("🔄 Criando dados de exemplo...")
        
        # Gerar 100 jogos de exemplo
        sample_data = []
        base_date = datetime(2020, 1, 1)
        
        for i in range(100):
            # Gerar 15 números únicos entre 1 e 25
            numbers = sorted(np.random.choice(range(1, 26), 15, replace=False))
            
            game_data = {
                'Concurso': i + 1,
                'DataSorteio': base_date + pd.Timedelta(days=i*3),
            }
              # Adicionar números no formato correto (N1 a N15)
            for j, num in enumerate(numbers, 1):
                game_data[f'N{j}'] = num                  # Adicionar algumas features básicas calculadas
            game_data['QtdePrimos'] = sum(1 for n in numbers if n in [2,3,5,7,11,13,17,19,23])
            game_data['QtdeFibonacci'] = sum(1 for n in numbers if n in [1,2,3,5,8,13,21])
            game_data['QtdeImpares'] = sum(1 for n in numbers if n % 2 == 1)
            game_data['SomaTotal'] = sum(numbers)
            game_data['Quintil1'] = sum(1 for n in numbers if 1 <= n <= 5)
            game_data['Quintil2'] = sum(1 for n in numbers if 6 <= n <= 10)
            game_data['Quintil3'] = sum(1 for n in numbers if 11 <= n <= 15)
            game_data['Quintil4'] = sum(1 for n in numbers if 16 <= n <= 20)
            game_data['Quintil5'] = sum(1 for n in numbers if 21 <= n <= 25)
                
            sample_data.append(game_data)
        
        df = pd.DataFrame(sample_data)
        print(f"✅ {len(df)} jogos de exemplo criados")
        return df
    
    def prepare_training_data(self, df: pd.DataFrame) -> tuple:
        """Prepara dados para treinamento do modelo com features avançadas"""
        print("🔧 Preparando dados para treinamento...")
          # Identificar colunas de números (N1 a N15)
        number_columns = [f'N{i}' for i in range(1, 16)]
        
        # Verificar se as colunas existem
        available_number_cols = [col for col in number_columns if col in df.columns]
        
        if not available_number_cols:
            # Fallback para formato antigo (Num1 a Num15)
            number_columns = [f'Num{i}' for i in range(1, 16)]
            available_number_cols = [col for col in number_columns if col in df.columns]
            
            if not available_number_cols:
                # Fallback para formato mais antigo
                number_columns = [col for col in df.columns if col.startswith('num_')]
                if not number_columns:
                    raise ValueError("Nenhuma coluna de números encontrada")
                available_number_cols = number_columns
        
        print(f"📊 Usando colunas de números: {available_number_cols}")
        
        # Criar matriz de números base
        X_numbers = df[available_number_cols].values
          # Adicionar features adicionais COMPLETAS se disponíveis
        feature_columns = [
            # Features estatísticas básicas
            'QtdePrimos', 'QtdeFibonacci', 'QtdeImpares', 'SomaTotal',
            
            # Distribuição por quintis (muito importante!)
            'Quintil1', 'Quintil2', 'Quintil3', 'Quintil4', 'Quintil5',
            
            # Padrões de gaps e repetições
            'QtdeGaps', 'QtdeRepetidos', 'SEQ', 'DistanciaExtremos',
            
            # Análise de sequências
            'ParesSequencia', 'QtdeMultiplos3', 'ParesSaltados',
            
            # Análise espacial/faixas
            'Faixa_Baixa', 'Faixa_Media', 'Faixa_Alta',
            'RepetidosMesmaPosicao',
            
            # Features antigas (fallback)
            'QtdePares', 'Media', 'Amplitude', 'DistanciaMedia', 'QtdeSequencia',
            'Col1', 'Col2', 'Col3', 'Col4', 'Col5',
            'Linha1', 'Linha2', 'Linha3', 'Linha4', 'Linha5',
            'DiagonalPrincipal', 'DiagonalSecundaria', 'Cruzeta'
        ]
        
        # Selecionar apenas features que existem no DataFrame
        available_features = [col for col in feature_columns if col in df.columns]
        
        if available_features:
            print(f"🎯 Usando {len(available_features)} features adicionais")
            X_features = df[available_features].values
            
            # Normalizar features separadamente
            from sklearn.preprocessing import StandardScaler
            scaler_features = StandardScaler()
            X_features_norm = scaler_features.fit_transform(X_features)
            
            # Combinar números normalizados com features
            X_numbers_norm = X_numbers / 25.0
            X_combined = np.concatenate([X_numbers_norm, X_features_norm], axis=1)
            
            print(f"📈 Matriz combinada: {X_combined.shape}")
            
        else:
            print("⚠️ Usando apenas números básicos (sem features avançadas)")
            X_combined = X_numbers / 25.0
        
        # Para predição sequencial, usar o jogo anterior para prever o próximo
        X_train = X_combined[:-1]  # Todos exceto o último
        y_train = X_numbers[1:] / 25.0   # Apenas números do próximo jogo, normalizados
        
        print(f"✅ Dados preparados:")
        print(f"   - Amostras de treinamento: {X_train.shape[0]}")
        print(f"   - Features de entrada: {X_train.shape[1]}")
        print(f"   - Números de saída: {y_train.shape[1]}")
        
        return X_train, y_train, X_numbers, available_number_cols

class LoterIAModel:
    """Modelo de Deep Learning para predição de loteria"""
      def __init__(self, config: LoterIAConfig):
        self.config = config
        self.model = None
        
    def build_model(self, input_shape: tuple) -> tf.keras.Model:
        """Constrói o modelo de rede neural OTIMIZADO"""
        print("🧠 Construindo modelo de IA avançado...")
        
        # Input layer
        inputs = tf.keras.layers.Input(shape=input_shape)
        
        # Primeira branch: Processamento dos números
        numbers_branch = tf.keras.layers.Dense(128, activation='relu')(inputs)
        numbers_branch = tf.keras.layers.BatchNormalization()(numbers_branch)
        numbers_branch = tf.keras.layers.Dropout(0.3)(numbers_branch)
        
        # Segunda branch: Processamento das features estatísticas
        features_branch = tf.keras.layers.Dense(64, activation='relu')(inputs)
        features_branch = tf.keras.layers.BatchNormalization()(features_branch)
        features_branch = tf.keras.layers.Dropout(0.2)(features_branch)
        
        # Concatenar branches
        combined = tf.keras.layers.Concatenate()([numbers_branch, features_branch])
        
        # Camadas densas profundas
        x = tf.keras.layers.Dense(256, activation='relu')(combined)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        # Camada de saída com ativação sigmóide para números normalizados
        outputs = tf.keras.layers.Dense(15, activation='sigmoid')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Compilar com otimizador mais sofisticado
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mape']  # Adicionar MAPE para melhor avaliação
        )
        
        print("✅ Modelo avançado construído com sucesso")
        print(f"📈 Parâmetros treináveis: {model.count_params():,}")
        
        return model
    
    def train(self, X_train, y_train, epochs=100, validation_split=0.2):
        """Treina o modelo"""
        print(f"🚀 Iniciando treinamento ({epochs} épocas)...")
        
        if self.model is None:
            self.model = self.build_model(X_train.shape[1:])
        
        # Callbacks para monitoramento
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
        ]
        
        # Treinar modelo
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Treinamento concluído!")
        return history
    
    def predict(self, X_input) -> np.ndarray:
        """Faz predições"""
        if self.model is None:
            raise ValueError("Modelo não foi treinado ainda")
        
        predictions = self.model.predict(X_input)
        
        # Desnormalizar e arredondar para números inteiros
        predictions_denorm = (predictions * 25).round().astype(int)
        
        # Garantir que estão no range válido
        predictions_denorm = np.clip(predictions_denorm, 1, 25)
        
        return predictions_denorm
    
    def save_model(self, filepath: str = None):
        """Salva o modelo treinado"""
        if filepath is None:
            filepath = os.path.join(self.config.MODEL_SAVE_PATH, "loteria_model.h5")
        
        if self.model:
            self.model.save(filepath)
            print(f"💾 Modelo salvo em: {filepath}")
        else:
            print("⚠️ Nenhum modelo para salvar")
    
    def load_model(self, filepath: str = None):
        """Carrega modelo salvo"""
        if filepath is None:
            filepath = os.path.join(self.config.MODEL_SAVE_PATH, "loteria_model.h5")
        
        if os.path.exists(filepath):
            self.model = tf.keras.models.load_model(filepath)
            print(f"📂 Modelo carregado de: {filepath}")
        else:
            print(f"❌ Arquivo não encontrado: {filepath}")

class LoterIAPredictor:
    """Sistema principal de predição"""
      def __init__(self):
        self.config = LoterIAConfig()
        self.config.ensure_directories()
        
        self.db_manager = DatabaseManager(self.config)
        self.data_processor = DataProcessor(self.db_manager)
        self.model = LoterIAModel(self.config)
        self.analyzer = PredictionAnalyzer(self.db_manager)  # Nova classe de análise
          def run_full_pipeline(self):
        """Executa pipeline COMPLETO com análises avançadas"""
        print("🎯 Iniciando LoterIA - Sistema de Predição AVANÇADO")
        print("=" * 60)
        
        try:
            # 1. Carregar dados históricos
            print("\n📊 FASE 1: Carregamento de Dados")
            print("-" * 40)
            df = self.data_processor.load_historical_data()
            
            # 2. Carregar combinações para análise
            print("\n🎲 FASE 2: Carregamento de Combinações")
            print("-" * 40)
            combinations_df = self.analyzer.load_combinations_table(limit=100000)
            
            # 3. Preparar dados para treinamento
            print("\n🔧 FASE 3: Preparação de Dados")
            print("-" * 40)
            X_train, y_train, X_numbers_raw, number_columns = self.data_processor.prepare_training_data(df)
            
            # 4. Treinar modelo com mais épocas
            print("\n🚀 FASE 4: Treinamento do Modelo")
            print("-" * 40)
            history = self.model.train(X_train, y_train, epochs=100, validation_split=0.2)
            
            # 5. Salvar modelo
            self.model.save_model()
            
            # 6. Gerar múltiplas predições
            print("\n🔮 FASE 5: Geração de Predições")
            print("-" * 40)
            
            # Gerar 5 predições diferentes usando entradas ligeiramente variadas
            all_predictions = []
            all_analyses = []
            
            for i in range(5):
                print(f"🎯 Gerando predição {i+1}/5...")
                
                # Usar últimos jogos com pequenas variações
                input_idx = -1 - i if len(X_train) > i else -1
                last_game = X_train[input_idx:input_idx+1]
                
                # Fazer predição
                prediction = self.model.predict(last_game)[0]
                
                # Garantir 15 números únicos
                unique_prediction = self._ensure_unique_numbers(prediction)
                all_predictions.append(unique_prediction)
                
                # Análise detalhada da predição
                analysis = self.analyzer.calculate_prediction_confidence(unique_prediction, df)
                all_analyses.append(analysis)
                
                print(f"   Números: {sorted(unique_prediction)}")
                print(f"   Confiança: {analysis['confidence_score']:.1%}")
            
            # 7. Análise comparativa com combinações
            print("\n📈 FASE 6: Análise Comparativa")
            print("-" * 40)
            
            best_prediction_idx = max(range(len(all_analyses)), 
                                    key=lambda i: all_analyses[i]['confidence_score'])
            best_prediction = all_predictions[best_prediction_idx]
            best_analysis = all_analyses[best_prediction_idx]
            
            print(f"🏆 MELHOR PREDIÇÃO (Confiança: {best_analysis['confidence_score']:.1%})")
            print(f"   Números: {sorted(best_prediction)}")
            
            # Encontrar combinações similares
            if not combinations_df.empty:
                similar_combinations = self.analyzer.find_similar_combinations(
                    best_prediction, combinations_df, top_n=3
                )
                
                if not similar_combinations.empty:
                    print(f"\n🔍 Combinações similares encontradas:")
                    for idx, row in similar_combinations.iterrows():
                        combo_numbers = [row[f'N{i}'] for i in range(1, 16)]
                        print(f"   ID {row['ID']}: {combo_numbers}")
            
            # 8. Relatório detalhado
            print("\n📋 FASE 7: Relatório Final")
            print("-" * 40)
            self._save_detailed_predictions(all_predictions, all_analyses, best_prediction_idx)
            
            print(f"\n✅ Pipeline AVANÇADO concluído com sucesso!")
            print(f"🎯 {len(all_predictions)} predições geradas")
            print(f"🏆 Melhor confiança: {best_analysis['confidence_score']:.1%}")
            
        except Exception as e:
            print(f"❌ Erro no pipeline: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            self.db_manager.disconnect()
    
    def _ensure_unique_numbers(self, prediction: np.ndarray, max_attempts=10) -> np.ndarray:
        """Garante que a predição tenha exatamente 15 números únicos"""
        unique_nums = np.unique(prediction)
        
        if len(unique_nums) == 15:
            return unique_nums
        
        # Se temos menos de 15, preencher com números próximos
        if len(unique_nums) < 15:
            available = set(range(1, 26)) - set(unique_nums)
            needed = 15 - len(unique_nums)
            
            # Escolher números que mantêm padrões similares
            additional = np.random.choice(list(available), needed, replace=False)
            result = np.concatenate([unique_nums, additional])
        else:
            # Se temos mais de 15, escolher os 15 mais prováveis
            # (baseado na confiança da rede neural)
            prob_scores = np.abs(prediction - 0.5)  # Quanto mais longe de 0.5, mais confiante
            top_indices = np.argsort(prob_scores)[-15:]
            result = prediction[top_indices]
        
        return np.sort(result).astype(int)
    
    def _save_detailed_predictions(self, all_predictions, all_analyses, best_idx):
        """Salva relatório detalhado de predições"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.config.RESULTS_PATH, f"relatorio_detalhado_{timestamp}.txt")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("🎯 LoterIA - Relatório Detalhado de Predições\n")
            f.write("=" * 60 + "\n")
            f.write(f"Data/Hora: {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}\n")
            f.write(f"Total de predições: {len(all_predictions)}\n\n")
            
            # Detalhes de cada predição
            for i, (pred, analysis) in enumerate(zip(all_predictions, all_analyses)):
                f.write(f"PREDIÇÃO {i+1} {'🏆' if i == best_idx else ''}\n")
                f.write("-" * 30 + "\n")
                f.write(f"Números: {sorted(pred)}\n")
                f.write(f"Confiança: {analysis['confidence_score']:.1%}\n")
                f.write(f"Soma total: {analysis['statistical_analysis']['soma_total']}\n")
                f.write(f"Primos: {analysis['statistical_analysis']['qtde_primos']}\n")
                f.write(f"Ímpares: {analysis['statistical_analysis']['qtde_impares']}\n")
                f.write(f"Distribuição quintis: {analysis['statistical_analysis']['distribuicao_quintis']}\n")
                
                for rec in analysis['recommendations']:
                    f.write(f"• {rec}\n")
                f.write("\n")
            
            # Resumo estatístico
            f.write("RESUMO ESTATÍSTICO\n")
            f.write("-" * 30 + "\n")
            confidences = [a['confidence_score'] for a in all_analyses]
            f.write(f"Confiança média: {np.mean(confidences):.1%}\n")
            f.write(f"Melhor confiança: {max(confidences):.1%}\n")
            f.write(f"Pior confiança: {min(confidences):.1%}\n")
            f.write(f"\nRecomendação: Use a predição {best_idx + 1} (melhor confiança)\n")
        
        print(f"💾 Relatório detalhado salvo em: {filepath}")
        
        # Também salvar predição simples para compatibilidade
        simple_filepath = os.path.join(self.config.RESULTS_PATH, f"predicao_{timestamp}.txt")
        best_prediction = all_predictions[best_idx]
        
        with open(simple_filepath, 'w', encoding='utf-8') as f:
            f.write(f"LoterIA - Predição gerada em {datetime.now()}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Números preditos: {sorted(best_prediction)}\n")
            f.write(f"Confiança: {all_analyses[best_idx]['confidence_score']:.1%}\n")
            f.write(f"Soma total: {sum(best_prediction)}\n")
        
        print(f"💾 Predição principal salva em: {simple_filepath}")
    
    def _save_predictions(self, prediction):
        """Salva predições em arquivo"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.config.RESULTS_PATH, f"predicao_{timestamp}.txt")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"LoterIA - Predição gerada em {datetime.now()}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Números preditos: {sorted(prediction)}\n")
            f.write(f"Soma total: {sum(prediction)}\n")
            f.write(f"Números pares: {sum(1 for x in prediction if x % 2 == 0)}\n")
            f.write(f"Números ímpares: {sum(1 for x in prediction if x % 2 == 1)}\n")
        
        print(f"💾 Predição salva em: {filepath}")

class PredictionAnalyzer:
    """Analisador avançado de predições usando tabela de combinações"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        
    def load_combinations_table(self, limit=50000) -> pd.DataFrame:
        """Carrega combinações da tabela COMBINACOES_LOTOFACIL"""
        print(f"🎯 Carregando {limit:,} combinações pré-processadas...")
        
        try:
            query = f"""
            SELECT TOP {limit}
                ID, N1, N2, N3, N4, N5, N6, N7, N8, N9, N10,
                N11, N12, N13, N14, N15,
                QtdePrimos, QtdeFibonacci, QtdeImpares, SomaTotal,
                Quintil1, Quintil2, Quintil3, Quintil4, Quintil5,
                QtdeGaps, QtdeRepetidos, SEQ, DistanciaExtremos,
                ParesSequencia, QtdeMultiplos3, ParesSaltados,
                Faixa_Baixa, Faixa_Media, Faixa_Alta
            FROM COMBINACOES_LOTOFACIL
            ORDER BY NEWID()  -- Ordem aleatória para variedade
            """
            
            if self.db_manager.config.DB_TYPE.lower() == "sqlserver":
                import urllib.parse
                
                params = urllib.parse.quote_plus(
                    f"DRIVER={{{self.db_manager.config.SQL_DRIVER}}};"
                    f"SERVER={self.db_manager.config.SQL_SERVER};"
                    f"DATABASE={self.db_manager.config.SQL_DATABASE};"
                    f"Trusted_Connection=yes;"
                )
                engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")
                
                df = pd.read_sql(query, engine)
                print(f"✅ {len(df):,} combinações carregadas")
                return df
            else:
                print("⚠️ Tabela de combinações disponível apenas no SQL Server")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Erro ao carregar combinações: {e}")
            return pd.DataFrame()
    
    def calculate_prediction_confidence(self, prediction: np.ndarray, historical_data: pd.DataFrame) -> dict:
        """Calcula confiança da predição baseada em padrões históricos"""
        print("📊 Analisando confiança da predição...")
        
        analysis = {
            'prediction': prediction.tolist(),
            'confidence_score': 0.0,
            'pattern_matches': [],
            'statistical_analysis': {},
            'recommendations': []
        }
        
        # Análise estatística básica
        pred_sum = int(np.sum(prediction))
        pred_primes = sum(1 for n in prediction if n in [2,3,5,7,11,13,17,19,23])
        pred_odds = sum(1 for n in prediction if n % 2 == 1)
        
        # Comparar com histórico
        if len(historical_data) > 0:
            hist_sums = historical_data['SomaTotal'].values if 'SomaTotal' in historical_data.columns else []
            hist_primes = historical_data['QtdePrimos'].values if 'QtdePrimos' in historical_data.columns else []
            hist_odds = historical_data['QtdeImpares'].values if 'QtdeImpares' in historical_data.columns else []
            
            # Calcular desvios
            if len(hist_sums) > 0:
                sum_percentile = np.percentile(hist_sums, 50)
                sum_deviation = abs(pred_sum - sum_percentile) / sum_percentile
                analysis['statistical_analysis']['sum_deviation'] = sum_deviation
                
                # Confiança baseada na proximidade com a mediana histórica
                confidence_sum = max(0, 1 - sum_deviation)
                analysis['confidence_score'] += confidence_sum * 0.4
            
            if len(hist_primes) > 0:
                primes_mode = np.bincount(hist_primes).argmax() if len(hist_primes) > 0 else 0
                primes_match = 1 if pred_primes == primes_mode else 0.5
                analysis['confidence_score'] += primes_match * 0.3
            
            if len(hist_odds) > 0:
                odds_mode = np.bincount(hist_odds).argmax() if len(hist_odds) > 0 else 0
                odds_match = 1 if pred_odds == odds_mode else 0.5
                analysis['confidence_score'] += odds_match * 0.3
        
        # Análise de distribuição por quintis
        quintil_dist = [
            sum(1 for n in prediction if 1 <= n <= 5),   # Quintil 1
            sum(1 for n in prediction if 6 <= n <= 10),  # Quintil 2
            sum(1 for n in prediction if 11 <= n <= 15), # Quintil 3
            sum(1 for n in prediction if 16 <= n <= 20), # Quintil 4
            sum(1 for n in prediction if 21 <= n <= 25)  # Quintil 5
        ]
        
        analysis['statistical_analysis'] = {
            'soma_total': pred_sum,
            'qtde_primos': pred_primes,
            'qtde_impares': pred_odds,
            'qtde_pares': 15 - pred_odds,
            'distribuicao_quintis': quintil_dist,
            'amplitude': int(max(prediction) - min(prediction)),
            'sequencias': self._count_sequences(prediction)
        }
        
        # Recomendações baseadas na análise
        if analysis['confidence_score'] > 0.7:
            analysis['recommendations'].append("🟢 Alta confiança - Padrão bem alinhado com histórico")
        elif analysis['confidence_score'] > 0.5:
            analysis['recommendations'].append("🟡 Confiança média - Alguns padrões divergem do histórico")
        else:
            analysis['recommendations'].append("🔴 Baixa confiança - Padrão muito divergente do histórico")
            
        return analysis
    
    def _count_sequences(self, numbers: np.ndarray) -> int:
        """Conta sequências consecutivas nos números"""
        sorted_nums = sorted(numbers)
        sequences = 0
        current_seq = 1
        
        for i in range(1, len(sorted_nums)):
            if sorted_nums[i] == sorted_nums[i-1] + 1:
                current_seq += 1
            else:
                if current_seq >= 2:
                    sequences += 1
                current_seq = 1
                
        if current_seq >= 2:
            sequences += 1
            
        return sequences
    
    def find_similar_combinations(self, prediction: np.ndarray, combinations_df: pd.DataFrame, top_n=5) -> pd.DataFrame:
        """Encontra combinações similares na tabela de combinações"""
        print(f"🔍 Buscando {top_n} combinações mais similares...")
        
        if combinations_df.empty:
            return pd.DataFrame()
        
        # Calcular similaridade baseada nas features estatísticas
        pred_features = self._extract_features(prediction)
        
        similarities = []
        for idx, row in combinations_df.iterrows():
            comb_features = {
                'QtdePrimos': row['QtdePrimos'],
                'QtdeFibonacci': row['QtdeFibonacci'],
                'QtdeImpares': row['QtdeImpares'],
                'SomaTotal': row['SomaTotal'],
                'Quintil1': row['Quintil1'],
                'Quintil2': row['Quintil2'],
                'Quintil3': row['Quintil3'],
                'Quintil4': row['Quintil4'],
                'Quintil5': row['Quintil5']
            }
            
            # Calcular distância euclidiana normalizada
            distance = 0
            for feature in pred_features:
                if feature in comb_features:
                    # Normalizar para features diferentes
                    if feature == 'SomaTotal':
                        weight = 1.0 / 100  # Normalizar soma
                    else:
                        weight = 1.0
                    distance += ((pred_features[feature] - comb_features[feature]) * weight) ** 2
            
            similarity = 1 / (1 + np.sqrt(distance))
            similarities.append((idx, similarity))
        
        # Ordenar por similaridade e pegar top_n
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_indices = [x[0] for x in similarities[:top_n]]
        
        return combinations_df.iloc[top_indices].copy()
    
    def _extract_features(self, numbers: np.ndarray) -> dict:
        """Extrai features estatísticas de um conjunto de números"""
        return {
            'QtdePrimos': sum(1 for n in numbers if n in [2,3,5,7,11,13,17,19,23]),
            'QtdeFibonacci': sum(1 for n in numbers if n in [1,2,3,5,8,13,21]),
            'QtdeImpares': sum(1 for n in numbers if n % 2 == 1),
            'SomaTotal': int(np.sum(numbers)),
            'Quintil1': sum(1 for n in numbers if 1 <= n <= 5),
            'Quintil2': sum(1 for n in numbers if 6 <= n <= 10),
            'Quintil3': sum(1 for n in numbers if 11 <= n <= 15),
            'Quintil4': sum(1 for n in numbers if 16 <= n <= 20),
            'Quintil5': sum(1 for n in numbers if 21 <= n <= 25)
        }
def main():
    """Função principal"""
    print("🚀 LoterIA - Sistema de Predição de Loteria")
    print("Versão 1.0.0 - Desenvolvido com TensorFlow")
    print()
    
    try:
        # Criar instância do preditor
        predictor = LoterIAPredictor()
        
        # Executar pipeline completo
        predictor.run_full_pipeline()
        
    except KeyboardInterrupt:
        print("\n⚠️ Execução interrompida pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro crítico: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
