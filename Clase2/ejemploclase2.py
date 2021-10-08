#Importar librerias
import os
import sys
#from numpy import linalg

# < menor que, > mayor que, == igual, != diferente, <= menor que o igual, <= mayor que o igual.

#Declarar variables - Python toma el tipo de variable del dato que está recibiendo, no es tipado.
nombre1 = "Hayde"
nombre2 = "Maria"
nombre3 = "Enrique"
numdealumnos = 13
numdealumnosenfloat = 13.0
alumnosqueentrarontarde = ["Alex", "Rafael", "Oscar"]
asistenciasclase = {"Primera Clase" : 13, "Segunda Clase" : 13}

#Imprimir los tipos de variables que tengo - ejemplo de programación estructurada
print("La variable nombre 1 es del tipo " + str(type(nombre1)))
print("La variable numdealumnos es del tipo " + str(type(numdealumnos)))
print("La variable numdealumnosenfloat del tipo " + str(type(numdealumnosenfloat)))
print("La variable alumnosqueentrarontarde es del tipo " + str(type(alumnosqueentrarontarde)))
print("La variable asistenciasclase es del tipo " + str(type(asistenciasclase)))

#Ejemplo de programación modular
#Definiendo mi función
def saludo(nombre):
    print("Hola "+nombre)

def solosaludositellamashaydeoconcepcion(nombre):
    if nombre == "Hayde" :
        print("Hola "+nombre)
    elif nombre == "Concepcion":
        print("Hola "+nombre)
    else:
        print("A ti no te saludo porque no te llamas Concepcion o Hayde" )

def imprimenombresalumnosqllt(listadenombres):
    for nombre in listadenombres:
        print(nombre + " llegó tarde")

def imprimenombresalumnosqllt2(listadenombres):
    for x in range(len(listadenombres)):
        print(listadenombres[x] + " llegó tarde")

    # contador = 0
    # while contador < len(listadenombres):
    #     print listadenombres[contador]

#Llamar a mi función
saludo(nombre1)
saludo(nombre2)
saludo(nombre3)
solosaludositellamashaydeoconcepcion(nombre1)
solosaludositellamashaydeoconcepcion(nombre2)
imprimenombresalumnosqllt(alumnosqueentrarontarde)
imprimenombresalumnosqllt2(alumnosqueentrarontarde)

#Ejemplo de programación orientada a objetos

class Saludo():
    def __init__(self):
        print("Estoy instanciando un objeto de la clase saludo\n")
    def saludo(self, nombre):
        print("Hola "+ nombre)

#instanciando un objeto de clase saludo
try:
    objetosaludo = Saludo()
except(e):
    pass #Ejemplo try catch
objetosaludo.saludo(nombre1)
objetosaludo.saludo(nombre2)
objetosaludo.saludo(nombre3)
