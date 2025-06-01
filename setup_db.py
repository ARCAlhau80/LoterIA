import sqlite3
import random
import os

print('Creating database...')

if os.path.exists('data/loteria.db'):
    os.remove('data/loteria.db')

conn = sqlite3.connect('data/loteria.db')
cursor = conn.cursor()

cursor.execute('''CREATE TABLE resultados (
    concurso INTEGER, 
    N1 INTEGER, N2 INTEGER, N3 INTEGER, N4 INTEGER, N5 INTEGER,
    N6 INTEGER, N7 INTEGER, N8 INTEGER, N9 INTEGER, N10 INTEGER,
    N11 INTEGER, N12 INTEGER, N13 INTEGER, N14 INTEGER, N15 INTEGER
)''')

for i in range(100):
    numbers = sorted(random.sample(range(1, 26), 15))
    cursor.execute('INSERT INTO resultados VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                   (1000+i, *numbers))

conn.commit()
cursor.execute('SELECT COUNT(*) FROM resultados')
print(f'Records created: {cursor.fetchone()[0]}')
conn.close()
print('Done!')
