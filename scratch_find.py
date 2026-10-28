with open('web/index.html', 'r', encoding='utf-8') as f:
    text = f.read()

pos = text.rfind('</html>')
print("End of web/index.html around </html>:")
print(text[pos-100:pos+300])
