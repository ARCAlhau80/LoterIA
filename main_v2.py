#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoterIA v2.0 - Sistema AVANÇADO de Predição de Loteria
Versão 2.0.0 - Integração com Combinações Pré-processadas
Desenvolvido com TensorFlow + SQL Server
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

# Machine Learning
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class LoterIAConfig:
    """Configurações globais do sistema"""
    DB_TYPE: str = "sqlite"  # sqlite ou sqlserver
    SQL_SERVER: str = "ROCHA1\\SQL2022"
    SQL_DATABASE: str = "LoterIA"
    SQL_DRIVER: str = "ODBC Driver 17 for SQL Server"
    SQLITE_PATH: str = "data/loteria.db"
    
    # Configurações do modelo
    MODEL_PATH: str = "models/loteria_model_v2.h5"
    RESULTS_PATH: str = "results/"
    
    # Parâmetros de treinamento
    EPOCHS: int = 50
    BATCH_SIZE: int = 32
    VALIDATION_SPLIT: float = 0.2
    LEARNING_RATE: float = 0.001

class DatabaseManager:
    """Gerenciador de conexões de banco de dados"""
    
    def __init__(self, config: LoterIAConfig):
        self.config = config
        self.connection = None
        
    def connect(self) -> bool:
        """Conecta ao banco de dados"""
        try:
            if self.config.DB_TYPE.lower() == "sqlserver":
                conn_str = (
                    f"DRIVER={{{self.config.SQL_DRIVER}}};"
                    f"SERVER={self.config.SQL_SERVER};"
                    f"DATABASE={self.config.SQL_DATABASE};"
                    f"Trusted_Connection=yes;"
                )
                self.connection = pyodbc.connect(conn_str)
                print("✅ Conectado ao SQL Server")
            else:
                os.makedirs(os.path.dirname(self.config.SQLITE_PATH), exist_ok=True)
                self.connection = sqlite3.connect(self.config.SQLITE_PATH)
                print("✅ Conectado ao SQLite")
            return True
        except Exception as e:
            print(f"❌ Erro na conexão: {e}")
            return False
    
    def close(self):
        """Fecha a conexão"""
        if self.connection:
            self.connection.close()
            print("🔌 Conexão fechada")

class DataProcessor:
    """Processador de dados para treinamento"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        
    def load_data(self, limit: int = 2000) -> pd.DataFrame:
        """Carrega dados históricos com features avançadas"""
        print("📊 Carregando dados históricos do SQL Server...")
        
        if not self.db_manager.connect():
            return pd.DataFrame()
        
        try:            # Query expandida com features avançadas
            if self.db_manager.config.DB_TYPE.lower() == "sqlserver":
                query = f"""
                SELECT TOP {limit}
                    Concurso, DataConcurso,
                    N1, N2, N3, N4, N5, N6, N7, N8, N9, N10, N11, N12, N13, N14, N15,
                    QtdePrimos, QtdeFibonacci, QtdeImpares, SomaTotal,
                    Quintil1, Quintil2, Quintil3, Quintil4, Quintil5,
                    QtdeGaps, QtdeRepetidos, SEQ, DistanciaExtremos,
                    ParesSequencia, QtdeMultiplos3, ParesSaltados,
                    Faixa_Baixa, Faixa_Media, Faixa_Alta, RepetidosMesmaPosicao
                FROM Resultados_INT 
                ORDER BY Concurso DESC
                """
                df = pd.read_sql(query, self.db_manager.connection)
            else:
                # Fallback para SQLite com query simplificada
                simple_query = f"""
                SELECT concurso, concurso as Concurso, '2020-01-01' as DataConcurso,
                       N1, N2, N3, N4, N5, N6, N7, N8, N9, N10, N11, N12, N13, N14, N15
                FROM resultados 
                ORDER BY concurso DESC 
                LIMIT {limit}
                """
                df = pd.read_sql(simple_query, self.db_manager.connection)
            
            print(f"✅ {len(df):,} registros carregados da tabela Resultados_INT")
            
            if len(df) > 0:
                print(f"📅 Período: {df['DataConcurso'].min()} até {df['DataConcurso'].max()}")
                print(f"🎲 Concursos: {df['Concurso'].min()} até {df['Concurso'].max()}")
            
            return df.sort_values('Concurso')  # Ordenar cronologicamente
            
        except Exception as e:
            print(f"❌ Erro ao carregar dados: {e}")
            return pd.DataFrame()
        finally:
            self.db_manager.close()

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara dados para treinamento"""
        print("🔧 Preparando dados para treinamento...")
        
        # Colunas de números
        number_columns = ['N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9', 'N10', 'N11', 'N12', 'N13', 'N14', 'N15']
        print(f"📊 Usando colunas de números: {number_columns}")
        
        # Extrair números
        X_numbers = df[number_columns].values
        
        # Features avançadas disponíveis
        feature_columns = [
            'QtdePrimos', 'QtdeFibonacci', 'QtdeImpares', 'SomaTotal',
            'Quintil1', 'Quintil2', 'Quintil3', 'Quintil4', 'Quintil5',
            'QtdeGaps', 'QtdeRepetidos', 'SEQ', 'DistanciaExtremos',
            'ParesSequencia', 'QtdeMultiplos3', 'ParesSaltados',
            'Faixa_Baixa', 'Faixa_Media', 'Faixa_Alta', 'RepetidosMesmaPosicao'
        ]
        
        # Selecionar apenas features que existem no DataFrame
        available_features = [col for col in feature_columns if col in df.columns]
        
        if available_features:
            print(f"🎯 Usando {len(available_features)} features avançadas:")
            for i, feat in enumerate(available_features):
                print(f"   {i+1:2d}. {feat}")
                
            X_features = df[available_features].values
            
            # Tratar valores infinitos e NaN
            X_features = np.nan_to_num(X_features, nan=0, posinf=0, neginf=0)
            
            # Normalizar features separadamente
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
        
        # Validar dados para NaN/infinito
        if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
            print("⚠️ Detectados valores NaN/infinito em X_train - corrigindo...")
            X_train = np.nan_to_num(X_train, nan=0, posinf=1, neginf=-1)
        
        if np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)):
            print("⚠️ Detectados valores NaN/infinito em y_train - corrigindo...")
            y_train = np.nan_to_num(y_train, nan=0, posinf=1, neginf=-1)
        
        print(f"✅ Dados preparados:")
        print(f"   - Amostras de treinamento: {X_train.shape[0]}")
        print(f"   - Features de entrada: {X_train.shape[1]}")
        print(f"   - Números de saída: {y_train.shape[1]}")
        print(f"   - Range X_train: [{X_train.min():.3f}, {X_train.max():.3f}]")
        print(f"   - Range y_train: [{y_train.min():.3f}, {y_train.max():.3f}]")
        
        return X_train, y_train

class LoterIAModel:
    """Modelo de deep learning para predição"""
    
    def __init__(self, config: LoterIAConfig):
        self.config = config
        self.model = None
        
    def build_model(self, input_shape: int) -> None:
        """Constrói modelo neural avançado"""
        print("🧠 Construindo modelo de IA avançado...")
        
        # Entrada
        inputs = keras.layers.Input(shape=(input_shape,))
        
        # Camadas densas com BatchNormalization
        x = keras.layers.Dense(128, activation='relu')(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.3)(x)
        
        x = keras.layers.Dense(64, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.2)(x)
        
        x = keras.layers.Dense(32, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        
        # Saída para 15 números
        outputs = keras.layers.Dense(15, activation='sigmoid')(x)
        
        # Criar modelo
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compilar
        optimizer = keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE)
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        print("✅ Modelo avançado construído com sucesso")
        print(f"📈 Parâmetros treináveis: {self.model.count_params():,}")
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> keras.callbacks.History:
        """Treina o modelo"""
        print(f"🚀 Iniciando treinamento avançado ({self.config.EPOCHS} épocas)...")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                self.config.MODEL_PATH,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Treinar
        history = self.model.fit(
            X_train, y_train,
            epochs=self.config.EPOCHS,
            validation_split=self.config.VALIDATION_SPLIT,
            callbacks=callbacks,
            batch_size=self.config.BATCH_SIZE,
            verbose=1
        )
        
        print("✅ Treinamento avançado concluído!")
        return history
    
    def predict(self, X_input: np.ndarray) -> np.ndarray:
        """Faz predições com pós-processamento inteligente"""
        if self.model is None:
            raise ValueError("Modelo não foi treinado ainda")
        
        # Garantir que entrada tem o formato correto
        if len(X_input.shape) == 1:
            X_input = X_input.reshape(1, -1)
        
        try:
            predictions = self.model.predict(X_input, verbose=0, batch_size=1)
        except Exception as e:
            print(f"❌ Erro na predição: {e}")
            print(f"📊 Shape da entrada: {X_input.shape}")
            raise
        
        # Desnormalizar e arredondar para números inteiros
        predictions_denorm = (predictions * 25).round().astype(int)
        
        # Garantir números válidos (1-25)
        predictions_denorm = np.clip(predictions_denorm, 1, 25)
        
        # Garantir que são únicos (remover duplicatas)
        unique_predictions = []
        for pred_row in predictions_denorm:
            unique_nums = []
            for num in pred_row:
                if num not in unique_nums:
                    unique_nums.append(num)
            
            # Se não temos 15 números únicos, preencher com números faltantes
            while len(unique_nums) < 15:
                for candidate in range(1, 26):
                    if candidate not in unique_nums:
                        unique_nums.append(candidate)
                        if len(unique_nums) == 15:
                            break
            
            unique_predictions.append(sorted(unique_nums[:15]))
        
        return np.array(unique_predictions)
    
    def save_model(self):
        """Salva o modelo treinado"""
        if self.model:
            os.makedirs(os.path.dirname(self.config.MODEL_PATH), exist_ok=True)
            self.model.save(self.config.MODEL_PATH)
            print(f"💾 Modelo avançado salvo em: {self.config.MODEL_PATH}")

class LoterIAPredictor:
    """Sistema principal de predição"""
    
    def __init__(self):
        self.config = LoterIAConfig()
        self.db_manager = DatabaseManager(self.config)
        self.data_processor = DataProcessor(self.db_manager)
        self.model = LoterIAModel(self.config)
        
        # Criar diretórios
        os.makedirs("models", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        print("📁 Diretório criado/verificado: models/")
        print("📁 Diretório criado/verificado: data/")
        print("📁 Diretório criado/verificado: results/")
        
    def run_full_pipeline(self):
        """Executa pipeline completo de treinamento e predição"""
        print("🎯 LoterIA v2.0 - Sistema de Predição AVANÇADO")
        print("=" * 56)
        
        # Fase 1: Carregamento de dados
        print("📊 FASE 1: Carregamento de Dados")
        print("-" * 40)
        df = self.data_processor.load_data()
        
        if df.empty:
            print("❌ Não foi possível carregar dados")
            return
        
        # Fase 2: Preparação de dados
        print("🔧 FASE 2: Preparação de Dados")
        print("-" * 40)
        X_train, y_train = self.data_processor.prepare_data(df)
        
        # Fase 3: Treinamento do modelo
        print("🚀 FASE 3: Treinamento do Modelo")
        print("-" * 40)
        self.model.build_model(X_train.shape[1])
        history = self.model.train(X_train, y_train)
        self.model.save_model()
          # Fase 4: Geração de predições
        print("🔮 FASE 4: Geração de Predições")
        print("-" * 40)
        
        # Verificar se temos dados suficientes para predição
        if len(X_train) < 1:
            print("❌ Dados insuficientes para geração de predições")
            return
        
        # Gerar múltiplas predições
        predictions = []
        for i in range(3):
            print(f"🎯 Gerando predição {i+1}/3...")
            
            # Usar últimos jogos com pequenas variações
            input_idx = max(0, len(X_train) - 1 - i)
            last_game = X_train[input_idx:input_idx+1]
            
            # Verificar se temos dados válidos
            if len(last_game) == 0:
                print(f"   ⚠️ Dados insuficientes para predição {i+1}")
                continue
            
            # Fazer predição
            try:
                prediction = self.model.predict(last_game)[0]
                # Converter para números de loteria válidos (1-25)
                lottery_numbers = []
                for val in prediction:
                    num = max(1, min(25, int(val * 25)))
                    if num not in lottery_numbers:
                        lottery_numbers.append(num)
                
                # Garantir 15 números únicos
                while len(lottery_numbers) < 15:
                    new_num = np.random.randint(1, 26)
                    if new_num not in lottery_numbers:
                        lottery_numbers.append(new_num)
                
                lottery_numbers = sorted(lottery_numbers[:15])
                predictions.append(lottery_numbers)
                print(f"   Predição {i+1}: {lottery_numbers}")
                
            except Exception as e:
                print(f"   ❌ Erro na predição {i+1}: {e}")
                continue
          # Salvar resultado se temos predições
        if predictions:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results/predicao_v2_{timestamp}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("🚀 LoterIA v2.0 - Predições Avançadas\n")
                f.write("=" * 50 + "\n")
                f.write(f"Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
                f.write(f"Modelo: {self.config.MODEL_PATH}\n")
                f.write(f"Dados de treinamento: {len(df)} registros\n\n")
                
                for i, pred in enumerate(predictions):
                    f.write(f"PREDIÇÃO {i+1}:\n")
                    f.write(f"{' '.join(f'{n:02d}' for n in pred)}\n\n")
            
            print(f"💾 Predições salvas em: {filename}")
            
            # Exibir resultado final
            print("\n🎊 RESULTADO FINAL")
            print("=" * 30)
            best_prediction = predictions[0]  # Primeira predição como principal
            print(f"🎲 Predição recomendada: {' '.join(f'{n:02d}' for n in best_prediction)}")
            print(f"📊 Total de números: {len(best_prediction)}")
            print(f"📈 Soma: {sum(best_prediction)}")
        else:
            print("❌ Nenhuma predição foi gerada com sucesso")

def main():
    """Função principal"""
    print("🚀 LoterIA v2.0 - Sistema AVANÇADO de Predição de Loteria")
    print("Versão 2.0.0 - Integração com Combinações Pré-processadas")
    print("Desenvolvido com TensorFlow + SQL Server")
    print()
    
    try:
        # Criar instância do preditor
        predictor = LoterIAPredictor()
        
        # Executar pipeline completo
        predictor.run_full_pipeline()
        
    except KeyboardInterrupt:
        print("\\n⚠️ Execução interrompida pelo usuário")
    except Exception as e:
        print(f"\\n❌ Erro crítico: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
