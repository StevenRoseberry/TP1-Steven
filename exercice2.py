import sympy as sp
import matplotlib.pyplot as plt
import numpy as np

# ** J'ai peu vérifié les réponses des mathématiques mais tout semble bon :) **

x, y, z = sp.symbols("x y z")

# le pprint c'est convenable non ? :)

# 1. f(x)
f = 3 * x ** 3 - 5 * x ** 2 + 2 * x - 7

print("-----------------")
print("Afficher dans la console mais beau")

# a)
sp.pprint(f)

print("-----------------")
print("Calculer la dérivée")

# b)
differentiel = sp.diff(f, x)
sp.pprint(differentiel)

print("-----------------")
print("Points critiques")

# c)
points_critiques = sp.solve(differentiel, x)
sp.pprint(points_critiques)

print("-----------------")
print("Valeurs critiques")

# d)
valeurs_critiques = [f.subs(x, pc) for pc in points_critiques]
sp.pprint(valeurs_critiques)

print("-----------------")
print("Intégrale indéfinie")

# e)
# ah oui c'est plus facile que dans mon cours de math tout ça !
integrale_indefinie = sp.integrate(f, x)
sp.pprint(integrale_indefinie)

print("-----------------")
print("Intégrale définie")

# f)
integrale_definie = sp.integrate(f, (x, 0, 2))
sp.pprint(integrale_definie)


# g)
f_lambdify = sp.lambdify(x, f, "numpy")
fprime_lambdify = sp.lambdify(x, differentiel, "numpy")

x_valeurs = np.linspace(-2, 3, 400)
y_f = f_lambdify(x_valeurs)
y_fprime = fprime_lambdify(x_valeurs)

plt.figure(figsize=(10, 6))
plt.plot(x_valeurs, y_f, label="f(x)", color="blue")
plt.plot(x_valeurs, y_fprime, label="f'(x)", color="red")

# h)
# Aide de ChatGPT + la librairie sympy pour comprendre la démarche
for points_critiques, valeur in zip(points_critiques, valeurs_critiques):
    plt.scatter(points_critiques, valeur, color="black")
    # le :.2f arrondi à 2 décimales sinon le texte prend tout l'espace, mention à mon ami ChatGPT
    plt.text(points_critiques, valeur, f"({points_critiques:.2f}, {valeur:.2f})", fontsize=9, ha="center")

# i)
x0 = -1
y0 = f.subs(x, x0)
pente = differentiel.subs(x, x0)
tangente = pente * (x - x0) + y0
tangente_lambd = sp.lambdify(x, tangente, "numpy")
plt.plot(x_valeurs, tangente_lambd(x_valeurs), "--", color="green", label="Tangente en x=-1")

plt.grid(True)
plt.legend()
plt.title("Graphes de f(x), f'(x), points critiques, tangente et tout le blabla")
plt.show()

print("-----------------")
print("Résolution du système")

# 2. système d'équation

# a)
systeme_eq = [
    sp.Eq(2 * x + 3 * y + z, 2),
    sp.Eq(-x + 2 * y + 3 * z, -1),
    sp.Eq(-3 * x - 3 * y + z, 0)
]

solution = sp.solve(systeme_eq, (x, y, z))
sp.pprint(solution)

print("-----------------")
print("Vérifications")

# b)
verifications = [equations.subs(solution) for equations in systeme_eq]
sp.pprint(verifications)

# 3. g(x)
print("-----------------")
print("Factorisation")

g = 6 * x ** 5 - 9 * x ** 4 - 49 * x ** 3 + 87 * x ** 2 - 17 * x + 30

# a)
g_factorise = sp.factor(g)
sp.pprint(g_factorise)

print("-----------------")
print("Développer pour vérifier")

# b)
g_developpe = sp.expand(g_factorise)
sp.pprint(g_developpe)

print("-----------------")
