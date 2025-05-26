import requests
import json

# URL of the API endpoint
url = "http://127.0.0.1:5000/api/optimizar-cursos"

# Set headers for JSON content type
headers = {'Content-Type': 'application/json'}

# Make the POST request with empty JSON body and proper headers
response = requests.post(url, json={}, headers=headers)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    resultado = response.json()
    
    # Print the results in a formatted way
    print("\n=== RESULTADO DE LA OPTIMIZACIÓN ===")
    print(f"Puntuación total máxima: {resultado['puntuacion_total']}")
    
    print("\n=== ASIGNACIÓN ÓPTIMA DE DÍAS ===")
    for asignacion in resultado['asignacion']:
        print(f"Curso {asignacion['curso']}: {asignacion['dias']} días")
    
    print("\n=== TABLA DE DECISIONES (DÍA A DÍA) ===")
    for decision in resultado['tabla_decisiones']:
        print(f"Día {decision['dia']}: Estudiar {decision['curso']} " +
              f"(día {decision['dias_acumulados']} de este curso) - " +
              f"Puntuación: {decision['puntuacion']}")
    
    print("\n=== PUNTUACIONES POR CURSO ===")
    for curso in resultado['puntuaciones_cursos']:
        print(f"Curso {curso['curso']}: {curso['dias_asignados']} días - " +
              f"Puntuación: {curso['puntuacion']}")
    
    print("\n=== CONTRIBUCIÓN PORCENTUAL A LA PUNTUACIÓN TOTAL ===")
    for curso, porcentaje in resultado['contribucion_porcentual'].items():
        print(f"Curso {curso}: {porcentaje}%")
    
    print("\n=== TABLA DE PUNTUACIONES ORIGINAL ===")
    print("Días | C1 | C2 | C3 | C4")
    for i, fila in enumerate(resultado['tabla_puntuaciones']):
        print(f"{i+1}    | {fila[0]} | {fila[1]} | {fila[2]} | {fila[3]}")
        
    print(f"\n=== INFORMACIÓN SOBRE EL PROCESO DE OPTIMIZACIÓN ===")
    print(f"Total de estados explorados: {resultado['total_estados_explorados']}")
    print(f"Descripción: {resultado['descripcion']}")
    
    print("\n=== PRIMEROS ESTADOS EXPLORADOS (MUESTRA) ===")
    for i, estado in enumerate(resultado['estados_explorados'][:5]):
        print(f"Estado {i+1}: Días restantes={estado['dias_restantes']}, " +
              f"Asignación={estado['asignados']}, " +
              f"Mejor puntuación={estado['mejor_puntuacion']}, " +
              f"Mejor decisión=C{estado['mejor_decision']+1 if estado['mejor_decision'] is not None else 'Ninguna'}")

else:
    print(f"Error en la solicitud: {response.status_code}")
    print(response.text)
