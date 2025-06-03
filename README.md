# 🎲 LoterIA - Sistema de Predição de Loteria com Deep Learning

## ✅ STATUS DO PROJETO (Junho 2025)

- **✅ TOTALMENTE FUNCIONAL** - Todas as versões testadas e operacionais
- **✅ CÓDIGO NO GITHUB** - Repositório: [ARCAlhau80/LoterIA](https://github.com/ARCAlhau80/LoterIA)
- **✅ BANCO DE DADOS** - 100 concursos de dados históricos carregados
- **✅ MODELOS TREINADOS** - Redes neurais v1.0 e v2.0 disponíveis
- **✅ ANÁLISE DE PADRÕES** - Sistema v3.0 com análise avançada implementado
- **✅ DEMO DISPONÍVEL** - Execute `python demo_improvements.py` para ver as funcionalidades
- **✅ TESTES VALIDADOS** - Todos os imports e funcionalidades verificados

## 🎯 INÍCIO RÁPIDO

```bash
# Testar funcionalidades v3.0
python demo_improvements.py

# Executar sistema completo v3.0
python main_v3.py

# Executar sistema avançado v2.0  
python main_v2.py

# Executar sistema básico v1.0
python main.py
```

## 🚀 NOVIDADES v3.0 ⭐

- **🔍 Análise de Divergência Posicional**: Números "quentes" e "frios" por posição
- **📊 Padrões Recorrentes**: Detecta e prevê reaparecimento de configurações
- **🎯 Combinações Inteligentes**: Gera jogos baseados em análise de padrões
- **🚨 Alertas Sazonais**: Avisa quando padrões estão próximos de se repetir
- **🧠 Predições Híbridas**: Combina IA neural + análise estatística

# LoterIA - Sistema Avançado de Predição de Loteria

🎲 **LoterIA v3.0** é um sistema avançado de predição de loteria que utiliza deep learning e técnicas de machine learning para analisar dados históricos da LOTOFACIL e gerar predições inteligentes.

## 🚀 Versões Disponíveis

### **Versão 1.0** (`main.py`)
- Sistema básico funcional
- Modelo neural simples
- Conectividade com SQLite/SQL Server
- Interface console básica

### **Versão 2.0** (`main_v2.py`)
- **Modelo Neural Avançado**: Arquitetura otimizada com BatchNormalization
- **Features Estatísticas Expandidas**: 20+ features avançadas (Quintis, Gaps, Sequências)
- **Sistema de Confiança**: Análise de confiança das predições
- **Múltiplas Predições**: Geração de várias predições com análise comparativa
- **Validação de Dados**: Tratamento robusto de NaN e valores infinitos
- **Callbacks Avançados**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

### **Versão 3.0** (`main_v3.py`) ⭐ **NOVA - RECOMENDADA**
- **🔍 Análise de Padrões Avançada**: Sistema completo de análise posicional
- **🎯 Divergência Posicional**: Identifica números "quentes" e "frios" por posição
- **📊 Padrões Recorrentes**: Detecta e prevê reaparecimento de padrões específicos
- **📈 Análise Temporal**: Compara frequências em janelas de tempo
- **🧠 Predições Híbridas**: Combina IA neural + análise de padrões
- **⚖️ Sistema de Pesos**: Pontuação inteligente baseada em divergências
- **🚨 Alertas Sazonais**: Detecta quando padrões estão próximos de se repetir

## 🛠️ Tecnologias

- **Python 3.8+**
- **TensorFlow 2.13+** - Framework de deep learning
- **pandas** - Manipulação e análise de dados
- **numpy** - Computação numérica
- **pyodbc** - Conectividade SQL Server
- **sqlite3** - Banco de dados local
- **scikit-learn** - Utilitários de machine learning

## 🔧 Instalação

1. **Criar ambiente virtual**:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

2. **Instalar dependências**:
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Como Usar

### **Execução Rápida - Versão 3.0** (Nova - Recomendada)
```bash
python main_v3.py
```

### **Demonstração das Melhorias**
```bash
python demo_improvements.py
```

### **Execução Versão 2.0** (Avançada)
```bash
python main_v2.py
```

### **Execução Versão 1.0** (Básica)
```bash
python main.py
```

## 🔥 Novas Funcionalidades v3.0

### **🔍 Pattern Analyzer** - Análise Avançada de Padrões
```python
from pattern_analyzer import PatternAnalyzer

analyzer = PatternAnalyzer()

# 1. Análise de Divergência Posicional
df_div = analyzer.analisar_divergencia_posicional(janela_size=15)

# 2. Padrões Recorrentes
padrao = {'numero_1': 1, 'numero_15': 25}
resultado = analyzer.analisar_padrao_posicional_recorrente(padrao)

# 3. Análise de Frequência Temporal
df_freq = analyzer.analisar_frequencia_temporal(frame_size=15)

# 4. Combinações Inteligentes
combinacoes = analyzer.gerar_combinacoes_inteligentes(df_div, limite=10)
```

### **🎯 Predições Híbridas** - IA + Padrões
- Combina redes neurais LSTM com análise de padrões
- Sistema de confiança baseado em múltiplas fontes
- Alertas sazonais para padrões recorrentes
- Métricas avançadas de tendência e divergência

### **📊 Análises Implementadas**
1. **Divergência Posicional**: Identifica números fora do padrão histórico
2. **Padrões Recorrentes**: Detecta e prevê reaparecimento de configurações específicas
3. **Frequência Temporal**: Compara janelas de tempo para identificar tendências
4. **Filtros Condicionais**: Análise estatística baseada em critérios específicos
5. **Pesos Inteligentes**: Sistema de pontuação baseado em análises múltiplas

## 📁 Estrutura do Projeto

```
LoterIA/
├── main.py                 # Versão 1.0 - Sistema básico
├── main_v2.py              # Versão 2.0 - Sistema avançado
├── main_v3.py              # Versão 3.0 - Sistema com padrões ⭐
├── pattern_analyzer.py     # Módulo de análise de padrões 🆕
├── demo_improvements.py    # Demonstração das melhorias 🆕
├── requirements.txt        # Dependências Python
├── README.md              # Esta documentação
├── data/                  # Dados e banco SQLite
│   └── loteria.db         # Base de dados local
├── models/                # Modelos treinados
│   ├── loteria_model.h5       # Modelo v1.0
│   ├── loteria_model_v2.h5    # Modelo v2.0
│   └── loteria_model_v3.h5    # Modelo v3.0 🆕
├── results/               # Resultados das predições
└── venv/                 # Ambiente virtual Python
```

## 🧠 Arquitetura do Modelo v2.0

### **Modelo Neural Avançado**
- **Camadas de Entrada**: Features combinadas (números + estatísticas)
- **Camadas Ocultas**: 128 → 64 → 32 neurônios
- **BatchNormalization**: Para estabilidade do treinamento
- **Dropout**: Para prevenção de overfitting
- **Saída**: 15 números preditos

### **Features Utilizadas**
1. **Básicas**: QtdePrimos, QtdeFibonacci, QtdeImpares, SomaTotal
2. **Quintis**: Quintil1, Quintil2, Quintil3, Quintil4, Quintil5
3. **Padrões**: QtdeGaps, QtdeRepetidos, SEQ, DistanciaExtremos
4. **Sequências**: ParesSequencia, QtdeMultiplos3, ParesSaltados
5. **Espaciais**: Faixa_Baixa, Faixa_Media, Faixa_Alta, RepetidosMesmaPosicao

## 📊 Processo de Execução

### **Versão 2.0 - Pipeline Completo**
1. **Carregamento de Dados**: 2000 registros históricos mais recentes
2. **Preparação de Dados**: Normalização e tratamento de NaN/infinitos
3. **Treinamento Avançado**: 50 épocas com callbacks inteligentes
4. **Geração de Predições**: 3 predições diferentes com análise
5. **Salvamento**: Resultados detalhados em arquivos timestamped

### **Melhorias da v2.0**
- ✅ **Dados Robustos**: Tratamento de valores NaN/infinito
- ✅ **Treinamento Inteligente**: EarlyStopping e ReduceLR
- ✅ **Validação Cruzada**: 20% dos dados para validação
- ✅ **Múltiplas Predições**: Análise comparativa
- ✅ **Logging Detalhado**: Acompanhamento completo do processo

## 🎲 Saída das Predições

### **Arquivo de Resultado** (`results/predicao_v2_YYYYMMDD_HHMMSS.txt`)
```
🚀 LoterIA v2.0 - Predições Avançadas
==================================================
Data/Hora: 01/06/2025 12:34:56
Modelo: models/loteria_model_v2.h5
Dados de treinamento: 2000 registros

PREDIÇÃO 1:
01 03 07 09 12 15 17 19 21 22 23 24 25

PREDIÇÃO 2:
02 04 06 08 11 13 16 18 20 21 23 24 25

PREDIÇÃO 3:
01 05 08 10 12 14 16 18 20 22 23 24 25
```

## ⚠️ Requisitos do Sistema

- **Python**: 3.8 ou superior
- **RAM**: Mínimo 4GB (8GB recomendado)
- **Espaço**: 1GB livre para modelos e dados
- **Conexão**: Para acesso ao SQL Server (opcional)

## 🔧 Configuração Avançada

### **Parâmetros Modificáveis** (em `LoterIAConfig`)
```python
DB_TYPE: str = "sqlite"          # sqlite ou sqlserver  
EPOCHS: int = 50                 # Épocas de treinamento
BATCH_SIZE: int = 32             # Tamanho do lote
VALIDATION_SPLIT: float = 0.2    # % para validação
LEARNING_RATE: float = 0.001     # Taxa de aprendizado
```

## 🚨 Solução de Problemas

### **Erros Comuns**
1. **ModuleNotFoundError**: Execute `pip install -r requirements.txt`
2. **TensorFlow não encontrado**: `pip install tensorflow==2.13.0`
3. **Erro de memória**: Reduza BATCH_SIZE para 16
4. **Conexão SQL**: Verifique string de conexão e drivers ODBC

### **Performance**
- **Treinamento lento**: Use GPU se disponível
- **Muitos dados**: Reduza limite de registros carregados
- **Predições imprecisas**: Aumente número de épocas

## 📈 Histórico de Versões

### **v2.0** (Atual)
- ✅ Modelo neural avançado
- ✅ Features estatísticas expandidas
- ✅ Sistema de múltiplas predições
- ✅ Tratamento robusto de dados
- ✅ Interface melhorada

### **v1.0**
- ✅ Sistema básico funcional
- ✅ Modelo neural simples
- ✅ Conectividade de banco

## ⚠️ Aviso Legal

Este sistema é destinado para fins educacionais e de pesquisa. As predições são baseadas em análise estatística e NÃO garantem resultados de apostas. Use com responsabilidade.

---

**Desenvolvido com TensorFlow + Python | LoterIA v2.0** 🚀
