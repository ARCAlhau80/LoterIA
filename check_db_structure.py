#!/usr/bin/env python3
"""
Script para verificar a estrutura da base de dados
"""

import sqlite3
import os

def check_database_structure():
    """Verifica a estrutura da base de dados"""
    db_path = 'data/loteria.db'
    
    if not os.path.exists(db_path):
        print(f"❌ Base de dados não encontrada: {db_path}")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Listar todas as tabelas
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print("📋 Tabelas encontradas:")
        for table in tables:
            print(f"  - {table[0]}")
        
        # Verificar estrutura de cada tabela
        for table in tables:
            table_name = table[0]
            print(f"\n🔍 Estrutura da tabela '{table_name}':")
            cursor.execute(f'PRAGMA table_info({table_name});')
            columns = cursor.fetchall()
            for col in columns:
                print(f"  {col[1]} ({col[2]})")
            
            # Mostrar alguns registos de exemplo
            cursor.execute(f'SELECT COUNT(*) FROM {table_name}')
            count = cursor.fetchone()[0]
            print(f"  📊 Total de registos: {count}")
            
            if count > 0:
                cursor.execute(f'SELECT * FROM {table_name} LIMIT 3')
                rows = cursor.fetchall()
                print(f"  📝 Primeiros 3 registos:")
                for i, row in enumerate(rows, 1):
                    print(f"    {i}: {row}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Erro ao verificar base de dados: {e}")

if __name__ == "__main__":
    check_database_structure()
