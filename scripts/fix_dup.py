import sys
p = 'app/streamlit_app.py'
t = open(p, encoding='utf-8').read()
positions = []
i = 0
while True:
    j = t.find('from __future__', i)
    if j < 0:
        break
    positions.append(j)
    i = j + 1
print('from __future__ positions:', positions)
print('total len:', len(t))
if len(positions) >= 2:
    new = t[:positions[1]]
    open(p, 'w', encoding='utf-8', newline='\n').write(new)
    print('rewritten, new len:', len(new))
