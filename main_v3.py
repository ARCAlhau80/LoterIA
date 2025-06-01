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
import logging
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from collections import Counter

# Machine Learning
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
    DB_TYPE: str = "sqlite"
    SQLITE_PATH: str = "data/loteria.db"
    
    # Configurações do modelo
    MODEL_PATH: str = "models/loteria_model_v3.h5"
    RESULTS_PATH: str = "results/"
    
    # Parâmetros de treinamento
    EPOCHS: int = 100
    BATCH_SIZE: int = 32
    VALIDATION_SPLIT: float = 0.2
    LEARNING_RATE: float = 0.001
    
    # Parâmetros de análise de padrões
    JANELA_DIVERGENCIA: int = 15
    JANELA_FREQUENCIA: int = 15
    MAX_COMBINACOES: int = 10

class DatabaseManager:
    """Gerenciador de conexões de banco de dados"""
    
    def __init__(self, config: LoterIAConfig):
        self.config = config
        self.connection = None
    
    def connect(self) -> sqlite3.Connection:
        """Estabelece conexão com o banco"""
        try:
            self.connection = sqlite3.connect(self.config.SQLITE_PATH)
            logger.info("✅ Conexão com banco de dados estabelecida")
            return self.connection
        except Exception as e:
            logger.error(f"❌ Erro ao conectar com banco: {e}")
            raise
    
    def close(self) -> None:
        """Fecha conexão com o banco"""
        if self.connection:
            self.connection.close()
            logger.info("🔌 Conexão com banco fechada")

class DataProcessor:
    """Processador de dados com recursos avançados"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.scaler = StandardScaler()
        self.pattern_analyzer = PatternAnalyzer(db_manager.config.SQLITE_PATH)
    
    def load_data(self) -> pd.DataFrame:
        """Carrega dados históricos"""
        try:
            conn = self.db_manager.connect()
            
            query = """
            SELECT * FROM resultados 
            ORDER BY concurso ASC
            """
            
            df = pd.read_sql_query(query, conn)
            logger.info(f"📊 Dados carregados: {len(df)} registros")
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
            numeros = [row[f'numero_{i}'] for i in range(1, 16)]
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
    """Modelo avançado de rede neural v3.0"""
    
    def __init__(self, config: LoterIAConfig):
        self.config = config
        self.model = None
        self.history = None
    
    def build_model(self, input_shape: Tuple[int, ...]) -> keras.Model:
        """Constrói modelo neural avançado"""
        logger.info(f"🏗️ Construindo modelo neural v3.0 com input_shape: {input_shape}")
        
        # Input layer
        inputs = keras.Input(shape=input_shape)
        
        # LSTM layers para análise temporal
        x = keras.layers.LSTM(128, return_sequences=True, dropout=0.2)(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LSTM(64, return_sequences=True, dropout=0.2)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LSTM(32, dropout=0.2)(x)
        
        # Dense layers com regularização
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.BatchNormalization()(x)
        
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.BatchNormalization()(x)
        
        x = keras.layers.Dense(64, activation='relu')(x)
        x = keras.layers.Dropout(0.1)(x)
        
        # Output layer - 15 números (1-25 normalizados)
        outputs = keras.layers.Dense(15, activation='sigmoid')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name="LoterIA_v3")
        
        # Compilar modelo
        optimizer = keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        self.model = model
        logger.info("✅ Modelo construído com sucesso")
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Treina o modelo com callbacks avançados"""
        logger.info("🚀 Iniciando treinamento do modelo...")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=self.config.MODEL_PATH,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Treinamento
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.EPOCHS,
            batch_size=self.config.BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("✅ Treinamento concluído")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Faz predição"""
        if self.model is None:
            logger.error("❌ Modelo não foi treinado")
            raise ValueError("Modelo não treinado")
        
        predictions = self.model.predict(X, verbose=0)
        return predictions
    
    def save(self, filepath: str = None) -> None:
        """Salva o modelo"""
        if filepath is None:
            filepath = self.config.MODEL_PATH
        
        if self.model:
            self.model.save(filepath)
            logger.info(f"💾 Modelo salvo em: {filepath}")
    
    def load(self, filepath: str = None) -> None:
        """Carrega modelo existente"""
        if filepath is None:
            filepath = self.config.MODEL_PATH
        
        if os.path.exists(filepath):
            self.model = keras.models.load_model(filepath)
            logger.info(f"📂 Modelo carregado de: {filepath}")
        else:
            logger.warning(f"⚠️ Arquivo de modelo não encontrado: {filepath}")

class LoterIAPredictorV3:
    """Sistema de predição avançado v3.0 com análise de padrões"""
    
    def __init__(self, config: LoterIAConfig):
        self.config = config
        self.db_manager = DatabaseManager(config)
        self.data_processor = DataProcessor(self.db_manager)
        self.model = LoterIAModelV3(config)
        self.pattern_analyzer = PatternAnalyzer(config.SQLITE_PATH)
    
    def train_model(self) -> None:
        """Treina o modelo completo"""
        logger.info("🎯 Iniciando processo de treinamento completo...")
        
        # 1. Carregar dados
        df = self.data_processor.load_data()
        
        # 2. Extrair features
        features = self.data_processor.extract_features_enhanced(df)
        
        # 3. Criar sequências
        X, y = self.data_processor.create_sequences(features)
        
        # 4. Dividir dados
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.config.VALIDATION_SPLIT, random_state=42
        )
        
        # 5. Construir modelo
        self.model.build_model(X_train.shape[1:])
        
        # 6. Treinar
        self.model.train(X_train, y_train, X_val, y_val)
        
        # 7. Salvar
        self.model.save()
        
        logger.info("🎉 Treinamento completo finalizado!")
    
    def predict_next_draw(self, num_predictions: int = 5) -> List[Dict]:
        """Gera predições inteligentes combinando IA + análise de padrões"""
        logger.info(f"🔮 Gerando {num_predictions} predições inteligentes...")
        
        try:
            # 1. Carregar modelo se necessário
            if self.model.model is None:
                self.model.load()
            
            # 2. Carregar dados para contexto
            df = self.data_processor.load_data()
            features = self.data_processor.extract_features_enhanced(df)
            
            # 3. Análise de padrões
            logger.info("🔍 Executando análise de padrões...")
            
            # Análise de divergência posicional
            df_divergencia = self.pattern_analyzer.analisar_divergencia_posicional(
                janela_size=self.config.JANELA_DIVERGENCIA
            )
            
            # Análise de frequência temporal
            df_frequencia = self.pattern_analyzer.analisar_frequencia_temporal(
                frame_size=self.config.JANELA_FREQUENCIA
            )
            
            # 4. Predições da IA
            logger.info("🤖 Gerando predições com IA...")
            
            # Usar últimas sequências para predição
            sequence_length = 10
            last_sequences = features[-sequence_length:]
            last_sequences = last_sequences.reshape(1, sequence_length, -1)
            
            ai_predictions = []
            for i in range(num_predictions):
                pred = self.model.predict(last_sequences)[0]
                # Desnormalizar predições (converter de 0-1 para 1-25)
                pred_numbers = (pred * 24) + 1
                pred_numbers = np.round(pred_numbers).astype(int)
                pred_numbers = np.clip(pred_numbers, 1, 25)
                ai_predictions.append(sorted(set(pred_numbers)))
            
            # 5. Combinações inteligentes baseadas em padrões
            logger.info("🎯 Gerando combinações baseadas em padrões...")
            
            df_combinacoes_padroes = self.pattern_analyzer.gerar_combinacoes_inteligentes(
                df_divergencia=df_divergencia,
                limite_combinacoes=num_predictions,
                usar_pesos=True
            )
            
            # 6. Combinar resultados
            predicoes_finais = []
            
            for i in range(num_predictions):
                predicao = {
                    'id': i + 1,
                    'metodo': 'IA + Padrões',
                    'confianca': 0.0,
                    'numeros': [],
                    'origem': 'hibrida'
                }
                
                # Combinar IA + Padrões se disponível
                if i < len(ai_predictions) and i < len(df_combinacoes_padroes):
                    ai_nums = set(ai_predictions[i])
                    pattern_nums = set(df_combinacoes_padroes.iloc[i][
                        [f'numero_{j}' for j in range(1, 16)]
                    ].values)
                    
                    # Média ponderada entre IA e padrões
                    combined_nums = list(ai_nums.union(pattern_nums))
                    if len(combined_nums) >= 15:
                        # Priorizar números que aparecem em ambos
                        intersection = ai_nums.intersection(pattern_nums)
                        remaining = list(set(combined_nums) - intersection)
                        np.random.shuffle(remaining)
                        
                        final_nums = list(intersection) + remaining[:15-len(intersection)]
                        predicao['numeros'] = sorted(final_nums[:15])
                        predicao['confianca'] = 0.8
                    else:
                        predicao['numeros'] = sorted(combined_nums)
                        predicao['confianca'] = 0.6
                        
                elif i < len(ai_predictions):
                    predicao['numeros'] = ai_predictions[i]
                    predicao['metodo'] = 'IA Neural'
                    predicao['confianca'] = 0.7
                    
                elif i < len(df_combinacoes_padroes):
                    pattern_nums = df_combinacoes_padroes.iloc[i][
                        [f'numero_{j}' for j in range(1, 16)]
                    ].values
                    predicao['numeros'] = sorted(pattern_nums)
                    predicao['metodo'] = 'Análise de Padrões'
                    predicao['confianca'] = 0.6
                
                # Garantir 15 números únicos
                while len(predicao['numeros']) < 15:
                    num_aleatorio = np.random.randint(1, 26)
                    if num_aleatorio not in predicao['numeros']:
                        predicao['numeros'].append(num_aleatorio)
                
                predicao['numeros'] = sorted(predicao['numeros'][:15])
                predicoes_finais.append(predicao)
            
            # 7. Adicionar análises complementares
            for predicao in predicoes_finais:
                self._add_analysis_metrics(predicao, df_frequencia, df_divergencia)
            
            logger.info(f"✅ {len(predicoes_finais)} predições geradas com sucesso")
            return predicoes_finais
            
        except Exception as e:
            logger.error(f"❌ Erro ao gerar predições: {e}")
            raise
    
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
    
    def save_predictions(self, predictions: List[Dict]) -> str:
        """Salva predições em arquivo"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"predicao_v3_{timestamp}.txt"
        filepath = os.path.join(self.config.RESULTS_PATH, filename)
        
        os.makedirs(self.config.RESULTS_PATH, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("LOTERIA v3.0 - PREDIÇÕES INTELIGENTES\n")
            f.write(f"Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            for pred in predictions:
                f.write(f"PREDIÇÃO #{pred['id']:02d}\n")
                f.write(f"Método: {pred['metodo']}\n")
                f.write(f"Confiança: {pred['confianca']:.1%}\n")
                f.write(f"Números: {' - '.join(f'{n:02d}' for n in pred['numeros'])}\n")
                
                if 'estatisticas' in pred:
                    stats = pred['estatisticas']
                    f.write(f"Soma Total: {stats['soma_total']}\n")
                    f.write(f"Pares/Ímpares: {stats['pares']}/{stats['impares']}\n")
                
                f.write("-" * 50 + "\n\n")
        
        logger.info(f"💾 Predições salvas em: {filepath}")
        return filepath

def main():
    """Função principal"""
    print("🎲 LoterIA v3.0 - Sistema Avançado de Predição de Loteria")
    print("=" * 60)
    
    try:
        # Configuração
        config = LoterIAConfig()
        
        # Inicializar sistema
        predictor = LoterIAPredictorV3(config)
        
        while True:
            print("\n📋 MENU PRINCIPAL:")
            print("1. 🚀 Treinar Modelo")
            print("2. 🔮 Gerar Predições")
            print("3. 🔍 Análise de Padrões")
            print("4. 📊 Análise de Divergência")
            print("5. 📈 Análise de Frequência Temporal")
            print("0. ❌ Sair")
            
            escolha = input("\n👉 Escolha uma opção: ").strip()
            
            if escolha == "1":
                print("\n🚀 Iniciando treinamento do modelo...")
                predictor.train_model()
                
            elif escolha == "2":
                num_pred = int(input("Quantas predições gerar? (1-10): ") or "5")
                predictions = predictor.predict_next_draw(num_pred)
                
                print(f"\n🔮 {len(predictions)} PREDIÇÕES GERADAS:")
                print("=" * 50)
                
                for pred in predictions:
                    print(f"\nPREDIÇÃO #{pred['id']:02d} - {pred['metodo']}")
                    print(f"Confiança: {pred['confianca']:.1%}")
                    print(f"Números: {' - '.join(f'{n:02d}' for n in pred['numeros'])}")
                    
                    if 'estatisticas' in pred:
                        stats = pred['estatisticas']
                        print(f"Soma: {stats['soma_total']} | P/I: {stats['pares']}/{stats['impares']}")
                
                # Salvar predições
                save_choice = input("\n💾 Salvar predições? (s/N): ").lower()
                if save_choice == 's':
                    filepath = predictor.save_predictions(predictions)
                    print(f"✅ Predições salvas em: {filepath}")
                    
            elif escolha == "3":
                print("\n🔍 Executando análise de padrões...")
                
                # Padrão personalizado
                print("Defina um padrão para análise (ex: posição 1 = número 1):")
                padrao = {}
                
                try:
                    pos = int(input("Posição (1-15): "))
                    num = int(input("Número (1-25): "))
                    padrao[f'numero_{pos}'] = num
                    
                    resultado = predictor.pattern_analyzer.analisar_padrao_posicional_recorrente(padrao)
                    if resultado:
                        print(f"\n📊 ANÁLISE DO PADRÃO:")
                        print(f"Ocorrências: {resultado['total_ocorrencias']}")
                        print(f"Média de distância: {resultado['media_distancia']:.1f}")
                        print(f"Última ocorrência: {resultado['ultima_ocorrencia']}")
                        print(f"Previsão: Concurso {resultado['concurso_previsto']} ± {resultado['margem_erro']}")
                        
                        if resultado['alerta_sazonal']:
                            print("🚨 ALERTA: Padrão próximo de se repetir!")
                    else:
                        print("⚠️ Padrão não encontrado nos dados históricos")
                        
                except ValueError:
                    print("❌ Valores inválidos")
                    
            elif escolha == "4":
                janela = int(input("Tamanho da janela para análise (padrão 15): ") or "15")
                
                print(f"\n📉 Executando análise de divergência (janela: {janela})...")
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
                
            elif escolha == "5":
                janela = int(input("Tamanho das janelas para comparação (padrão 15): ") or "15")
                
                print(f"\n📈 Executando análise de frequência temporal...")
                df_freq = predictor.pattern_analyzer.analisar_frequencia_temporal(janela)
                
                print("\n📊 TENDÊNCIAS DOS NÚMEROS:")
                print(df_freq[['numero', 'diferenca', 'tendencia']].head(15).to_string(index=False))
                
            elif escolha == "0":
                print("\n👋 Encerrando LoterIA v3.0...")
                break
                
            else:
                print("❌ Opção inválida!")
                
    except KeyboardInterrupt:
        print("\n\n⏹️ Programa interrompido pelo usuário")
    except Exception as e:
        logger.error(f"❌ Erro fatal: {e}")
        print(f"❌ Erro: {e}")

if __name__ == "__main__":
    main()
