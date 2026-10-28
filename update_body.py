file_path = 'scripts/build_frontend.py'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('<body class="authenticated">', '<body>', 1)

# Ensure initAuthGate correctly checks sessionStorage
# Let's inspect initAuthGate in content
if 'sessionStorage.getItem("darknode_access")' in content:
    print("sessionStorage check exists in initAuthGate!")

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Updated body tag to allow sign in page to display!")

