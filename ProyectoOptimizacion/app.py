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

if __name__ == '__main__':
    app.run(debug=True)