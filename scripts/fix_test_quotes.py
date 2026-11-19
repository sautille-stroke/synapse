with open("/root/work/dervgen/synapse/tests/test_property_contracts.py", "r", encoding="utf-8") as f:
    s = f.read()

s = s.replace("defect_id= def-01", 'defect_id="def-01"')
s = s.replace("category=mesh_crumple", 'category="mesh_crumple"')

with open("/root/work/dervgen/synapse/tests/test_property_contracts.py", "w", encoding="utf-8") as f:
    f.write(s)

print("Quotes fixed.")
