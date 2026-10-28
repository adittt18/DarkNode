with open('scripts/build_frontend.py', 'r', encoding='utf-8') as f:
    text = f.read()

pos = text.find('async function generateNewGraph()')
pos2 = text.find('catch', pos)
print(text[pos2:pos2+1200])
