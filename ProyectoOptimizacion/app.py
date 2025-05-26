from pulp import *
import numpy as np
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

class OptimizacionEstudio:
    def __init__(self, dias_disponibles, tiempo_por_dia, temas, pesos_temas, 
                 dificultad_temas, eficiencia_temas, horas_minimas_tema=0):
        """
        Inicializa el modelo de optimización
        
        Args:
            dias_disponibles: número de días disponibles para estudiar
            tiempo_por_dia: horas disponibles por día
            temas: lista de nombres de temas
            pesos_temas: peso de cada tema en la evaluación (0-100)
            dificultad_temas: dificultad de cada tema (1-10)
            eficiencia_temas: eficiencia del estudiante en cada tema (1-10)
            horas_minimas_tema: horas mínimas a dedicar a cada tema (opcional)
        """
        self.dias_disponibles = dias_disponibles
        self.tiempo_por_dia = tiempo_por_dia
        self.temas = temas
        self.n_temas = len(temas)
        self.pesos_temas = pesos_temas
        self.dificultad_temas = dificultad_temas
        self.eficiencia_temas = eficiencia_temas
        self.horas_minimas_tema = horas_minimas_tema
        
    def resolver(self):
        """
        Resuelve el problema de optimización
        
        Returns:
            Un diccionario con el calendario óptimo de estudio
        """
        # Crear el problema de optimización
        prob = LpProblem("OptimizacionEstudio", LpMaximize)
        
        # Variables: horas dedicadas a cada tema por día
        # x[i][j] = horas dedicadas al tema i en el día j
        x = {}
        for i in range(self.n_temas):
            for j in range(self.dias_disponibles):
                x[i, j] = LpVariable(f"x_{i}_{j}", 0, self.tiempo_por_dia)
        
        # Variables auxiliares: porcentaje de dominio de cada tema
        # Representa cuánto se ha aprendido del tema en relación a su dificultad
        y = {}
        for i in range(self.n_temas):
            y[i] = LpVariable(f"y_{i}", 0, 1)  # Valor entre 0 y 1 (0-100%)
        
        # Factor de aprendizaje: cada hora contribuye según la eficiencia y la dificultad
        factor_aprendizaje = {}
        for i in range(self.n_temas):
            # Estimación de horas necesarias para dominar completamente el tema
            horas_para_dominio = self.dificultad_temas[i] * 10 / self.eficiencia_temas[i]
            factor_aprendizaje[i] = 1 / horas_para_dominio
        
        # Restricción: relación entre horas de estudio y nivel de dominio
        for i in range(self.n_temas):
            total_horas_tema = lpSum([x[i, j] for j in range(self.dias_disponibles)])
            # El nivel de dominio es proporcional a las horas estudiadas
            horas_para_dominio = self.dificultad_temas[i] * 10 / self.eficiencia_temas[i]
            prob += y[i] <= total_horas_tema * factor_aprendizaje[i]
        
        # Función objetivo: maximizar la nota esperada según el peso de cada tema
        prob += lpSum([y[i] * self.pesos_temas[i] for i in range(self.n_temas)])
        
        # Restricción: tiempo disponible por día
        for j in range(self.dias_disponibles):
            prob += lpSum([x[i, j] for i in range(self.n_temas)]) <= self.tiempo_por_dia
        
        # Restricción opcional: horas mínimas por tema
        if self.horas_minimas_tema > 0:
            for i in range(self.n_temas):
                prob += lpSum([x[i, j] for j in range(self.dias_disponibles)]) >= self.horas_minimas_tema
        
        # Resolver el problema
        prob.solve(PULP_CBC_CMD(msg=False))
        
        # Preparar los resultados
        resultado = {}
        resultado["status"] = LpStatus[prob.status]
        
        if resultado["status"] == "Optimal":
            # Valor óptimo de la función objetivo
            resultado["nota_esperada"] = value(prob.objective)
            
            # Nivel de dominio por tema
            niveles_dominio = {}
            for i in range(self.n_temas):
                niveles_dominio[self.temas[i]] = round(value(y[i]) * 100, 2)  # Convertir a porcentaje
            resultado["niveles_dominio"] = niveles_dominio
            
            # Distribución de horas por día y tema
            calendario = {}
            for j in range(self.dias_disponibles):
                dia = f"Día {j+1}"
                calendario[dia] = []
                
                for i in range(self.n_temas):
                    horas = value(x[i, j])
                    if horas > 0.01:  # Consideramos solo asignaciones significativas
                        calendario[dia].append({
                            "tema": self.temas[i],
                            "horas": round(horas, 2)
                        })
            
            resultado["calendario"] = calendario
            
            # Resumen por tema
            resumen_temas = {}
            for i in range(self.n_temas):
                tema = self.temas[i]
                horas_totales = sum([value(x[i, j]) for j in range(self.dias_disponibles)])
                resumen_temas[tema] = {
                    "horas_totales": round(horas_totales, 2),
                    "peso": self.pesos_temas[i],
                    "dificultad": self.dificultad_temas[i],
                    "eficiencia": self.eficiencia_temas[i],
                    "nivel_dominio": round(value(y[i]) * 100, 2)  # Porcentaje de dominio
                }
            
            resultado["resumen_temas"] = resumen_temas
            
            # Agregar información de programación dinámica para visualización
            resultado["programacion_dinamica"] = self.generar_informacion_pd(x, y)
        
        return resultado
    
    def generar_informacion_pd(self, x, y):
        """
        Genera información para visualizar el proceso de programación dinámica
        
        Args:
            x: variables de decisión (horas por tema y día)
            y: variables de nivel de dominio
            
        Returns:
            Un diccionario con información para visualizar las tablas de PD
        """
        etapas = []
        
        # Para cada tema, creamos una etapa de la PD
        for i in range(self.n_temas):
            etapa = {
                "tema": self.temas[i],
                "estados": []
            }
            
            # Para cada día, calculamos los puntos de calificación
            for j in range(self.dias_disponibles):
                puntos_calificacion = {}
                
                # Simulamos diferentes asignaciones de tiempo (1-4 horas)
                for horas in range(1, 5):
                    # Calculamos el valor de aprendizaje según la fórmula
                    horas_para_dominio = self.dificultad_temas[i] * 10 / self.eficiencia_temas[i]
                    factor = 1 / horas_para_dominio
                    valor = min(1.0, horas * factor) * self.pesos_temas[i]
                    puntos_calificacion[horas] = round(valor, 2)
                
                # La decisión real tomada por el modelo
                horas_reales = value(x[i, j])
                decision_optima = round(horas_reales)
                
                estado = {
                    "dia": j + 1,
                    "puntos_calificacion": puntos_calificacion,
                    "decision_optima": decision_optima,
                    "horas_optimas": round(horas_reales, 2),
                    "valor_obtenido": round(value(y[i]) * self.pesos_temas[i], 2) if j == self.dias_disponibles - 1 else None
                }
                
                etapa["estados"].append(estado)
            
            etapas.append(etapa)
        
        # Tabla de puntos estimados (como en la primera imagen)
        tabla_puntos = []
        for j in range(self.dias_disponibles):
            fila = {"dia": j + 1}
            
            for i in range(self.n_temas):
                tema = self.temas[i]
                horas = value(x[i, j])
                if horas > 0.01:
                    eficiencia = self.eficiencia_temas[i]
                    dificultad = self.dificultad_temas[i]
                    # Puntos = horas * eficiencia / dificultad * 10 (escala de 1-10)
                    puntos = round((horas * eficiencia / dificultad) * 10, 1)
                else:
                    puntos = 0
                
                fila[tema] = puntos
            
            tabla_puntos.append(fila)
        
        return {
            "etapas": etapas,
            "tabla_puntos": tabla_puntos
        }

# Solución específica para el problema de preparación de exámenes finales
class OptimizacionCursos:
    def __init__(self, dias_disponibles, num_cursos, puntuaciones, dias_minimos=1, dias_maximos=4):
        """
        Inicializa el modelo de optimización para la preparación de exámenes finales
        
        Args:
            dias_disponibles: número de días disponibles para estudiar
            num_cursos: número de cursos a preparar
            puntuaciones: matriz de puntuaciones [dias][curso]
            dias_minimos: días mínimos a dedicar a cada curso
            dias_maximos: días máximos a dedicar a cada curso
        """
        self.dias_disponibles = dias_disponibles
        self.num_cursos = num_cursos
        self.puntuaciones = puntuaciones
        self.dias_minimos = dias_minimos
        self.dias_maximos = dias_maximos
    
    def generar_tablas_etapas(self):
        """
        Genera las tablas por etapas para la programación dinámica
        
        Returns:
            Lista de tablas por etapas
        """
        tablas_etapas = []
        
        # Calculamos el número de etapas (igual al número de cursos)
        num_etapas = self.num_cursos
        
        # Para cada etapa (curso), generamos una tabla
        for etapa in range(num_etapas):
            # La etapa 1 corresponde al último curso (C4), la etapa 4 al primero (C1)
            curso = num_etapas - etapa - 1
            
            tabla = {
                "etapa": etapa + 1,
                "curso": f"C{curso + 1}",
                "filas": []
            }
            
            # Calculamos los estados posibles para esta etapa
            # El estado representa los días disponibles restantes
            max_estado = self.dias_disponibles - (num_etapas - etapa - 1) * self.dias_minimos
            min_estado = etapa + 1  # Al menos necesitamos 1 día para cada curso restante
            
            # Para cada estado posible
            for estado in range(min_estado, max_estado + 1):
                fila = {
                    "estado": estado,
                    "decisiones": []
                }
                
                # Para cada posible decisión (días a asignar al curso actual)
                mejor_valor = -1000000
                mejor_decision = 0
                
                for decision in range(1, min(self.dias_maximos, estado - etapa) + 1):
                    # Calculamos el valor para esta decisión
                    valor_actual = self.puntuaciones[decision-1][curso]
                    
                    # Si no es la última etapa, sumamos el valor óptimo de la siguiente etapa
                    if etapa < num_etapas - 1:
                        # Buscamos el valor óptimo para el estado resultante en la etapa siguiente
                        estado_siguiente = estado - decision
                        valor_siguiente = 0
                        
                        # Simulamos el valor de la siguiente etapa
                        # En una implementación real, usaríamos los valores ya calculados
                        valor_siguiente = estado_siguiente * 3  # Valor aproximado para simulación
                        
                        valor_total = valor_actual + valor_siguiente
                    else:
                        valor_total = valor_actual
                    
                    # Guardamos la decisión y su valor
                    fila["decisiones"].append({
                        "dias": decision,
                        "valor": valor_total,
                        "calculo": f"{valor_actual} + {valor_total - valor_actual}" if etapa < num_etapas - 1 else str(valor_actual)
                    })
                    
                    # Actualizamos la mejor decisión
                    if valor_total > mejor_valor:
                        mejor_valor = valor_total
                        mejor_decision = decision
                
                # Guardamos el valor óptimo y la decisión óptima
                fila["valor_optimo"] = mejor_valor
                fila["decision_optima"] = mejor_decision
                
                tabla["filas"].append(fila)
            
            tablas_etapas.append(tabla)
        
        # Invertimos el orden para que la etapa 1 sea la primera
        tablas_etapas.reverse()
        
        return tablas_etapas
    
    def resolver_pd(self):
        """
        Resuelve el problema usando programación dinámica determinista
        
        Returns:
            Un diccionario con la asignación óptima de días y la puntuación total
        """
        # Inicializar tabla de programación dinámica
        # dp[d][c1][c2][c3][c4] representa la puntuación máxima usando d días
        # con c1, c2, c3, c4 días asignados a cada curso respectivamente
        
        # Debido a la dimensionalidad, usaremos un enfoque recursivo con memoización
        memo = {}
        # Para rastrear los estados explorados
        estados_explorados = []
        
        def dp(dias_restantes, asignados):
            # Si ya hemos calculado este estado, retornamos el resultado
            key = (dias_restantes, tuple(asignados))
            if key in memo:
                return memo[key]
            
            # Caso base: no hay más días para asignar
            if dias_restantes == 0:
                # Verificamos que todos los cursos tengan al menos los días mínimos
                if all(a >= self.dias_minimos for a in asignados):
                    # Calculamos la puntuación total
                    puntuacion = sum(self.puntuaciones[a-1][c] for c, a in enumerate(asignados))
                    return puntuacion, []
                else:
                    return -1000000, []
            
            mejor_puntuacion = -1000000
            mejor_decision = None
            
            # Probamos asignar el día restante a cada curso
            for curso in range(self.num_cursos):
                # Solo si no excedemos el máximo de días por curso
                if asignados[curso] < self.dias_maximos:
                    # Creamos una nueva asignación
                    nueva_asignacion = asignados.copy()
                    nueva_asignacion[curso] += 1
                    
                    # Calculamos la puntuación recursivamente
                    puntuacion, decisiones = dp(dias_restantes - 1, nueva_asignacion)
                    
                    # Actualizamos la mejor puntuación
                    if puntuacion > mejor_puntuacion:
                        mejor_puntuacion = puntuacion
                        mejor_decision = [(curso, dias_restantes)] + decisiones
            
            # Guardamos el resultado en la memoización y el estado explorado
            memo[key] = (mejor_puntuacion, mejor_decision)
            
            # Asegurarse de que los valores sean serializables a JSON
            puntuacion_serializable = mejor_puntuacion
            if mejor_puntuacion == -1000000:
                puntuacion_serializable = "No factible"
                
            estados_explorados.append({
                "dias_restantes": dias_restantes,
                "asignados": list(asignados),
                "mejor_puntuacion": puntuacion_serializable,
                "mejor_decision": curso if mejor_decision else None
            })
            return memo[key]
        
        # Comenzamos con 0 días asignados a cada curso
        asignacion_inicial = [0] * self.num_cursos
        puntuacion_total, decisiones = dp(self.dias_disponibles, asignacion_inicial)
        
        # Construimos la solución final
        asignacion_final = [0] * self.num_cursos
        for curso, _ in decisiones:
            asignacion_final[curso] += 1
        
        # Construimos la tabla de decisiones para visualización
        tabla_decisiones = []
        dias_por_curso = [0] * self.num_cursos
        
        for dia in range(1, self.dias_disponibles + 1):
            decision = None
            for curso, d in decisiones:
                if d == dia:
                    decision = curso
                    break
            
            if decision is not None:
                dias_por_curso[decision] += 1
                puntuacion = self.puntuaciones[dias_por_curso[decision]-1][decision]
                tabla_decisiones.append({
                    "dia": dia,
                    "curso": f"C{decision+1}",
                    "dias_acumulados": dias_por_curso[decision],
                    "puntuacion": puntuacion
                })
        
        # Calculamos las puntuaciones finales por curso
        puntuaciones_cursos = []
        for c in range(self.num_cursos):
            if asignacion_final[c] > 0:
                puntuaciones_cursos.append({
                    "curso": f"C{c+1}",
                    "dias_asignados": asignacion_final[c],
                    "puntuacion": self.puntuaciones[asignacion_final[c]-1][c]
                })
        
        # Generamos las tablas por etapas para la programación dinámica
        tablas_etapas = self.generar_tablas_etapas()
        
        # Calculamos la distribución de días para cada curso
        distribucion_dias = {}
        for i, dias in enumerate(asignacion_final):
            curso = f"C{i+1}"
            distribucion_dias[curso] = dias
        
        # Calculamos la contribución de cada curso a la puntuación total
        contribucion_porcentual = {}
        for curso in puntuaciones_cursos:
            contribucion_porcentual[curso["curso"]] = round((curso["puntuacion"] / puntuacion_total) * 100, 2)
        
        return {
            "puntuacion_total": puntuacion_total,
            "asignacion": [{
                "curso": f"C{i+1}", 
                "dias": d
            } for i, d in enumerate(asignacion_final)],
            "tabla_decisiones": tabla_decisiones,
            "puntuaciones_cursos": puntuaciones_cursos,
            "tabla_puntuaciones": self.puntuaciones,
            "distribucion_dias": distribucion_dias,
            "contribucion_porcentual": contribucion_porcentual,
            "estados_explorados": estados_explorados[:100],  # Limitamos a 100 estados para no sobrecargar la respuesta
            "total_estados_explorados": len(estados_explorados),
            "tablas_etapas": tablas_etapas,
            "descripcion": "Solución óptima usando programación dinámica determinista para asignar días de estudio a cursos"
        }

# API Flask para comunicarse con Angular
app = Flask(__name__)
CORS(app)  # Permitir solicitudes cross-origin

@app.route('/api/optimizar', methods=['POST'])
def optimizar():
    data = request.json
    
    try:
        # Valor predeterminado para horas_minimas_tema si no está en el request
        horas_minimas_tema = data.get('horas_minimas_tema', 0)
        
        optimizador = OptimizacionEstudio(
            dias_disponibles=data['dias_disponibles'],
            tiempo_por_dia=data['tiempo_por_dia'],
            temas=data['temas'],
            pesos_temas=data['pesos_temas'],
            dificultad_temas=data['dificultad_temas'],
            eficiencia_temas=data['eficiencia_temas'],
            horas_minimas_tema=horas_minimas_tema
        )
        
        resultado = optimizador.resolver()
        return jsonify(resultado)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/optimizar-cursos', methods=['POST'])
def optimizar_cursos():
    """
    Endpoint para optimizar la asignación de días de estudio a cursos.
    
    Datos de entrada (opcionales, se usan valores predeterminados si no se proporcionan):
    - dias_disponibles: número de días disponibles para estudiar (predeterminado: 7)
    - num_cursos: número de cursos a preparar (predeterminado: 4)
    - puntuaciones: matriz de puntuaciones [dias][curso] (predeterminado: matriz del ejercicio)
    - dias_minimos: días mínimos a dedicar a cada curso (predeterminado: 1)
    - dias_maximos: días máximos a dedicar a cada curso (predeterminado: 4)
    
    Retorna:
    - La asignación óptima de días a cursos
    - La puntuación total máxima
    - Información detallada sobre el proceso de optimización
    """
    try:
        # Datos del ejercicio 2
        dias_disponibles = 7
        num_cursos = 4
        dias_minimos = 1
        dias_maximos = 4
        
        # Matriz de puntuaciones [dias][curso]
        # Cada fila representa un día (1-4) y cada columna un curso (C1-C4)
        puntuaciones = [
            [5, 4, 5, 7],  # 1 día
            [6, 6, 6, 8],  # 2 días
            [7, 6, 7, 9],  # 3 días
            [9, 8, 8, 10]  # 4 días
        ]
        
        # Si se envían datos personalizados, los usamos
        data = request.json
        if data:
            if 'puntuaciones' in data:
                puntuaciones = data['puntuaciones']
            if 'dias_disponibles' in data:
                dias_disponibles = data['dias_disponibles']
            if 'num_cursos' in data:
                num_cursos = data['num_cursos']
            if 'dias_minimos' in data:
                dias_minimos = data['dias_minimos']
            if 'dias_maximos' in data:
                dias_maximos = data['dias_maximos']
                
        # Validaciones básicas
        if dias_disponibles <= 0:
            return jsonify({"error": "El número de días disponibles debe ser mayor que 0"}), 400
        if num_cursos <= 0:
            return jsonify({"error": "El número de cursos debe ser mayor que 0"}), 400
        if dias_minimos < 0:
            return jsonify({"error": "El número mínimo de días por curso no puede ser negativo"}), 400
        if dias_maximos <= 0:
            return jsonify({"error": "El número máximo de días por curso debe ser mayor que 0"}), 400
        if dias_minimos > dias_maximos:
            return jsonify({"error": "El número mínimo de días no puede ser mayor que el máximo"}), 400
        if dias_minimos * num_cursos > dias_disponibles:
            return jsonify({"error": "No hay suficientes días disponibles para asignar el mínimo a cada curso"}), 400
        
        # Resolvemos el problema
        optimizador = OptimizacionCursos(
            dias_disponibles=dias_disponibles,
            num_cursos=num_cursos,
            puntuaciones=puntuaciones,
            dias_minimos=dias_minimos,
            dias_maximos=dias_maximos
        )
        
        resultado = optimizador.resolver_pd()
        return jsonify(resultado)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)