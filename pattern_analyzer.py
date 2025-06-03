"""
LoterIA - Pattern Analyzer (CORRIGIDO)
Módulo avançado de análise de padrões baseado no script original
Integra análise de divergência posicional, padrões recorrentes e filtros condicionais
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
from itertools import combinations, product
from collections import Counter
from typing import Dict, List, Tuple, Optional
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatternAnalyzer:
    """Analisador avançado de padrões para LoterIA"""
    
    def __init__(self, db_path: str = "data/loteria.db", use_sqlserver_data: bool = False, db_manager=None):
        self.db_path = db_path
        self.use_sqlserver_data = use_sqlserver_data
        self.db_manager = db_manager
        self.df_historico = None
        self.load_historical_data()
    
    def load_historical_data(self) -> None:
        """Carrega dados históricos do banco"""
        try:
            if self.use_sqlserver_data and self.db_manager:
                # CORREÇÃO: Usar dados atualizados do SQL Server
                logger.info("📊 Carregando dados do SQL Server para análise de padrões...")
                conn = self.db_manager.connect()
                
                if self.db_manager.config.DB_TYPE.lower() == "sqlserver":
                    query = """
                    SELECT 
                        Concurso as concurso,
                        N1 as n1, N2 as n2, N3 as n3, N4 as n4, N5 as n5,
                        N6 as n6, N7 as n7, N8 as n8, N9 as n9, N10 as n10,
                        N11 as n11, N12 as n12, N13 as n13, N14 as n14, N15 as n15
                    FROM Resultados_INT 
                    ORDER BY Concurso ASC
                    """
                    self.df_historico = pd.read_sql_query(query, conn)
                    logger.info(f"✅ Dados atualizados carregados: {len(self.df_historico)} concursos do SQL Server")
                else:
                    # Fallback para SQLite se SQL Server não disponível
                    self._load_from_sqlite()
            else:
                # Método original: carregar do SQLite
                self._load_from_sqlite()
                
        except Exception as e:
            logger.error(f"❌ Erro ao carregar dados: {e}")
            # Fallback para SQLite
            self._load_from_sqlite()
    
    def _load_from_sqlite(self) -> None:
        """Método auxiliar para carregar do SQLite"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = """
            SELECT * FROM resultados 
            ORDER BY concurso ASC
            """
            self.df_historico = pd.read_sql_query(query, conn)
            conn.close()
            logger.info(f"✅ Dados históricos carregados: {len(self.df_historico)} concursos")
        except Exception as e:
            logger.error(f"❌ Erro ao carregar dados: {e}")
            raise
    
    def analisar_divergencia_posicional(self, janela_size: int = 15) -> pd.DataFrame:
        """
        Análise de divergência posicional - compara frequência histórica vs janela recente
          Args:
            janela_size: Tamanho da janela para análise recente
            
        Returns:
            DataFrame com análise de divergência por número/posição
        """
        logger.info(f"🔍 Iniciando análise de divergência posicional (janela: {janela_size})")
        
        df_total = self.df_historico.copy()
        df_janela = self.df_historico.tail(janela_size)
        
        posicoes = [f'n{i}' for i in range(1, 16)]  # Corrigido para minúsculas
        resultado = []
        
        for numero in range(1, 26):
            for pos in posicoes:
                # Frequência total histórica
                total_ocorrencias = (df_total[pos] == numero).sum()
                total_pct = total_ocorrencias / len(df_total) * 100
                
                # Frequência na janela recente
                janela_ocorrencias = (df_janela[pos] == numero).sum()
                janela_pct = janela_ocorrencias / len(df_janela) * 100
                
                # Divergência
                delta = janela_pct - total_pct
                status = self._avaliar_status_divergencia(delta)
                
                resultado.append({
                    'numero': numero,
                    'posicao': pos,
                    'freq_total_pct': round(total_pct, 2),
                    'freq_janela_pct': round(janela_pct, 2),
                    'divergencia_pct': round(delta, 2),
                    'status': status,
                    'peso_sugerido': self._calcular_peso_divergencia(janela_pct, status)
                })
        
        df_resultado = pd.DataFrame(resultado)
        
        # Log dos principais achados
        desvios_positivos = df_resultado[df_resultado['status'].isin(['🔥 Muito acima', '⬆️ Acima'])]
        desvios_negativos = df_resultado[df_resultado['status'].isin(['❄️ Muito abaixo', '⬇️ Abaixo'])]
        
        logger.info(f"📈 Desvios positivos encontrados: {len(desvios_positivos)}")
        logger.info(f"📉 Desvios negativos encontrados: {len(desvios_negativos)}")
        
        return df_resultado
    
    def _avaliar_status_divergencia(self, delta: float) -> str:
        """Avalia o status da divergência"""
        if delta > 15:
            return '🔥 Muito acima'
        elif delta > 5:
            return '⬆️ Acima'
        elif delta < -15:
            return '❄️ Muito abaixo'
        elif delta < -5:
            return '⬇️ Abaixo'
        else:
            return '✅ Estável'
    
    def _calcular_peso_divergencia(self, freq_janela: float, status: str) -> float:
        """Calcula peso baseado na divergência"""
        pesos = {
            '🔥 Muito acima': 0.1,
            '⬆️ Acima': 0.2,
            '✅ Estável': 0.5,
            '⬇️ Abaixo': 0.8,
            '❄️ Muito abaixo': 1.0
        }
        return freq_janela * pesos.get(status, 0.5)
    
    def analisar_padrao_posicional_recorrente(self, padroes: Dict[str, int]) -> Dict:
        """
        Análise de padrões posicionais recorrentes
        
        Args:
            padroes: Dicionário com posições e valores (ex: {'N1': 1, 'N15': 25})
            
        Returns:
            Dicionário com análise do padrão
        """
        logger.info(f"🔍 Analisando padrão recorrente: {padroes}")
        
        # Localizar concursos que respeitam o padrão
        mask = self.df_historico.apply(
            lambda row: all(row.get(pos) == val for pos, val in padroes.items()), 
            axis=1
        )
        concursos_match = self.df_historico.loc[mask, 'concurso'].tolist()
        
        if not concursos_match:
            logger.warning("⚠️ Nenhum concurso encontrado com o padrão informado")
            return None
        
        # Calcular estatísticas
        distancias = np.diff(concursos_match)
        media_dist = np.mean(distancias) if len(distancias) > 0 else 0
        mediana_dist = np.median(distancias) if len(distancias) > 0 else 0
        
        # Dados atuais
        concurso_atual = self.df_historico['concurso'].max()
        ultima_ocorrencia = concursos_match[-1]
        dist_atual = concurso_atual - ultima_ocorrencia
        
        # Alerta sazonal
        alerta = media_dist > 0 and dist_atual >= (media_dist * 0.9)
        
        # Previsão
        concurso_previsto = int(ultima_ocorrencia + media_dist) if media_dist > 0 else ultima_ocorrencia
        margem = max(1, int(media_dist * 0.1))
        
        # Estatísticas dos concursos com padrão
        estatisticas_cols = ['soma_total', 'qtde_pares', 'qtde_impares']
        # Adicionar colunas se existirem
        available_cols = [col for col in estatisticas_cols if col in self.df_historico.columns]
        
        if available_cols:
            estatisticas = self.df_historico[
                self.df_historico['concurso'].isin(concursos_match)
            ][available_cols].mean()
        else:
            estatisticas = pd.Series()
        
        resultado = {
            'concursos_ocorrencia': concursos_match,
            'total_ocorrencias': len(concursos_match),
            'distancias': distancias.tolist(),
            'media_distancia': media_dist,
            'mediana_distancia': mediana_dist,
            'ultima_ocorrencia': ultima_ocorrencia,
            'concurso_atual': concurso_atual,
            'distancia_atual': dist_atual,
            'alerta_sazonal': alerta,
            'concurso_previsto': concurso_previsto,
            'margem_erro': margem,
            'padrao': list(padroes.values()),
            'estatisticas_medias': estatisticas.to_dict() if not estatisticas.empty else {}
        }
        
        logger.info(f"📊 Padrão encontrado em {len(concursos_match)} concursos")
        logger.info(f"📅 Última ocorrência: {ultima_ocorrencia}, Previsão: {concurso_previsto} ± {margem}")
        
        if alerta:
            logger.warning("🚨 ALERTA: Padrão próximo de se repetir!")
        
        return resultado
    
    def diagnostico_filtros_condicionais(self, 
                                       coluna_filtro: str,
                                       valores_filtro: List[int],
                                       colunas_analise: List[str]) -> pd.DataFrame:
        """
        Diagnóstico de filtros condicionais - analisa estatísticas baseadas em filtro fixo
        
        Args:
            coluna_filtro: Nome da coluna para filtro
            valores_filtro: Lista de valores para filtrar
            colunas_analise: Colunas para análise estatística
            
        Returns:
            DataFrame com estatísticas condicionais
        """
        logger.info(f"🎯 Diagnóstico condicional: {coluna_filtro} in {valores_filtro}")
        
        # Filtrar dados
        df_filtrado = self.df_historico[
            self.df_historico[coluna_filtro].isin(valores_filtro)
        ]
        
        if df_filtrado.empty:
            logger.warning(f"⚠️ Nenhum registro encontrado para filtro {coluna_filtro} in {valores_filtro}")
            return pd.DataFrame()
        
        # Calcular estatísticas apenas para colunas existentes
        colunas_disponiveis = [col for col in colunas_analise if col in df_filtrado.columns]
        
        if not colunas_disponiveis:
            logger.warning(f"⚠️ Nenhuma coluna de análise disponível: {colunas_analise}")
            return pd.DataFrame()
        
        estatisticas = df_filtrado[colunas_disponiveis].describe().T
        
        # Adicionar moda
        try:
            modas = df_filtrado[colunas_disponiveis].mode()
            if not modas.empty:
                estatisticas['moda'] = modas.iloc[0]
        except Exception as e:
            logger.warning(f"⚠️ Erro ao calcular moda: {e}")
        
        logger.info(f"📊 Análise baseada em {len(df_filtrado)} concursos")
        
        return estatisticas
    
    def analisar_frequencia_temporal(self, frame_size: int = 15) -> pd.DataFrame:
        """
        Análise de frequência temporal - compara duas janelas de tempo
        
        Args:
            frame_size: Tamanho de cada janela
            
        Returns:
            DataFrame com análise de tendência
        """
        logger.info(f"📈 Analisando tendência temporal (janelas de {frame_size})")
        
        if len(self.df_historico) < frame_size * 2:
            logger.error(f"❌ Dados insuficientes. Necessário: {frame_size * 2}, Disponível: {len(self.df_historico)}")
            return pd.DataFrame()
        
        # Separar janelas
        df_passado = self.df_historico.tail(frame_size * 2).head(frame_size)
        df_atual = self.df_historico.tail(frame_size)
        
        def contar_frequencias(df_window):
            freq = {}
            posicoes = [f'n{i}' for i in range(1, 16)]  # Corrigido para minúsculas
            for col in posicoes:
                if col in df_window.columns:
                    for val in df_window[col].values:
                        freq[val] = freq.get(val, 0) + 1
            return freq
        
        freq_passado = contar_frequencias(df_passado)
        freq_atual = contar_frequencias(df_atual)
        
        # Construir resultado
        dados = []
        for numero in range(1, 26):
            atual = freq_atual.get(numero, 0)
            passado = freq_passado.get(numero, 0)
            media_atual = atual / frame_size if frame_size > 0 else 0
            media_passada = passado / frame_size if frame_size > 0 else 0
            diferenca = atual - passado
            
            # Classificar tendência
            if diferenca > 2:
                tendencia = '🔥 Forte alta'
            elif diferenca > 0:
                tendencia = '⬆️ Alta'
            elif diferenca < -2:
                tendencia = '❄️ Forte baixa'
            elif diferenca < 0:
                tendencia = '⬇️ Baixa'
            else:
                tendencia = '✅ Estável'
            
            dados.append({
                'numero': numero,
                'freq_atual': atual,
                'freq_passada': passado,
                'media_atual_pct': f'{media_atual:.1%}',
                'media_passada_pct': f'{media_passada:.1%}',
                'diferenca': diferenca,
                'tendencia': tendencia
            })
        
        df_resultado = pd.DataFrame(dados)
        df_resultado = df_resultado.sort_values(by='diferenca', ascending=False).reset_index(drop=True)
        
        # Log das principais tendências
        altas = df_resultado[df_resultado['tendencia'].isin(['🔥 Forte alta', '⬆️ Alta'])]
        baixas = df_resultado[df_resultado['tendencia'].isin(['❄️ Forte baixa', '⬇️ Baixa'])]
        
        logger.info(f"📈 Números em alta: {len(altas)}")
        logger.info(f"📉 Números em baixa: {len(baixas)}")
        
        return df_resultado
    
    def gerar_combinacoes_inteligentes(self, 
                                     df_divergencia: pd.DataFrame,
                                     limite_combinacoes: int = 10,
                                     usar_pesos: bool = True) -> pd.DataFrame:
        """
        Gera combinações baseadas na análise de divergência posicional
        
        Args:
            df_divergencia: DataFrame da análise de divergência
            limite_combinacoes: Número máximo de combinações
            usar_pesos: Se deve usar sistema de pesos
            
        Returns:
            DataFrame com combinações geradas
        """
        logger.info(f"🎯 Gerando {limite_combinacoes} combinações inteligentes")
        
        posicoes = [f'n{i}' for i in range(1, 16)]  # Corrigido para minúsculas
        
        # Mapear pesos por posição
        mapa_pesos = {pos: {} for pos in posicoes}
        
        for _, row in df_divergencia.iterrows():
            numero = int(row['numero'])
            posicao = row['posicao']
            peso = row['peso_sugerido'] if usar_pesos else 1.0
            
            if posicao in mapa_pesos:
                mapa_pesos[posicao][numero] = peso
        
        # Selecionar candidatos por posição (top 3)
        candidatos_por_posicao = {}
        for pos in posicoes:
            if pos in mapa_pesos and mapa_pesos[pos]:
                candidatos = sorted(
                    mapa_pesos[pos].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:3]
                candidatos_por_posicao[pos] = [num for num, peso in candidatos]
            else:
                # Fallback para números mais comuns
                candidatos_por_posicao[pos] = list(range(1, 26))[:3]
        
        # Gerar combinações
        possibilidades = [candidatos_por_posicao[pos] for pos in posicoes]
        combinacoes_cruas = list(product(*possibilidades))
        
        # Filtrar combinações válidas (15 números únicos)
        combinacoes_unicas = []
        for comb in combinacoes_cruas:
            numeros_unicos = sorted(set(comb))
            if len(numeros_unicos) == 15:
                combinacoes_unicas.append(numeros_unicos)
        
        # Limitar e pontuar
        combinacoes_unicas = list(set(tuple(c) for c in combinacoes_unicas))
        
        if usar_pesos:
            def pontuar(comb):
                score = 0
                for i, num in enumerate(comb):
                    pos = f'N{i+1}'
                    if pos in mapa_pesos and num in mapa_pesos[pos]:
                        score += mapa_pesos[pos][num]
                return score
            
            combinacoes_unicas = sorted(
                combinacoes_unicas, 
                key=pontuar, 
                reverse=True
            )
        
        # Limitar resultado
        combinacoes_finais = combinacoes_unicas[:limite_combinacoes]
        
        # Criar DataFrame
        df_resultado = pd.DataFrame(
            combinacoes_finais, 
            columns=posicoes
        )
        
        # Adicionar metadados
        df_resultado['data_geracao'] = datetime.now()
        df_resultado['metodo'] = 'divergencia_posicional'
        
        logger.info(f"✅ {len(df_resultado)} combinações geradas com sucesso")
        
        return df_resultado
    
    def salvar_analise(self, 
                      df_resultado: pd.DataFrame, 
                      nome_tabela: str = "analise_padroes") -> None:
        """Salva resultado da análise no banco"""
        try:
            conn = sqlite3.connect(self.db_path)
            df_resultado.to_sql(nome_tabela, conn, if_exists='replace', index=False)
            conn.close()
            logger.info(f"✅ Análise salva na tabela {nome_tabela}")
        except Exception as e:
            logger.error(f"❌ Erro ao salvar análise: {e}")
            raise

if __name__ == "__main__":
    # Exemplo de uso
    analyzer = PatternAnalyzer()
    
    # 1. Análise de divergência posicional
    print("🔍 Executando análise de divergência posicional...")
    df_divergencia = analyzer.analisar_divergencia_posicional(janela_size=15)
    print(df_divergencia.head(10))
    
    # 2. Análise de padrão recorrente
    print("\n🔍 Executando análise de padrão recorrente...")
    padrao_exemplo = {'N1': 1, 'N15': 25}
    resultado_padrao = analyzer.analisar_padrao_posicional_recorrente(padrao_exemplo)
    if resultado_padrao:
        print(f"Padrão encontrado em {resultado_padrao['total_ocorrencias']} concursos")
    
    # 3. Gerar combinações inteligentes
    print("\n🎯 Gerando combinações inteligentes...")
    df_combinacoes = analyzer.gerar_combinacoes_inteligentes(
        df_divergencia, 
        limite_combinacoes=5
    )
    print(df_combinacoes)
