# FIUBA-nx

Análisis de los datos del FIUBA-Map.

---

**Qué es esto?**

El [FIUBA-Map](https://fede.dm/FIUBA-Map/) es una herramienta de FIUBA para que los alumnos puedan ir marcando su carrera facultativa (que materias cursaron, que nota se sacaron, etc).

Si bien no es una red social (en el sentido criollo de "red social"), no deja de ser una herramienta que contiene información de personas y de como estas personas se relacionan entre sí. Estos datos no son directos: en el FMap en ningún momento marcas tu relación con otros alumnos (no existe botón de "Yo hice un trabajo práctico con X"), pero si pueden empezar a inferirse (vos y yo hacemos la misma carrera, vos y ella cursaron la misma materia, etc). De eso se trata este trabajo práctico para la materia de redes sociales! Encontrar y descubrir las relaciones implicitas entre distintos alumnos de la misma carrera.

El trabajo es más que nada un análisis exploratorio de que datos tenemos y como se relacionan entre sí, donde podemos ir utilizando conceptos teóricos de redes sociales, y concluye en poder responderle 3 preguntas a un alumno:

- Con qué otros alumnos puedo hacer TPs en un futuro?
- Qué materias electivas me conviene cursar?
- Cómo puedo organizar mi siguiente cuatrimestre?

---

**Cómo funciona?**

El trabajo se divide en la infra y los grafos:

- `infra.ipynb`: donde se levantan los datos del FMap, se parsean y se convierten en un dataframe de Pandas que pueda ser leído por NetworkX, para poder armar un grafo en base a eso.
- `{grafazo,graphito,grafon}.ipynb`: para cada una de las preguntas planteadas queremos ver que datos tenemos que puedan responderlas, que grafo podemos armar que nos ayude a contestarla, y efectivamente lograr contestarla.

---

**Cómo lo corro localmente para poder ver que electivas me convienen, mis compañeros de TP, y demás?**

Primero, instalar Poetry y setupear el repo.

```zsh
# (En algun virtualenv)
poetry install
poetry run pre-commit install
poetry run jupyter lab
```

Ahora si:

- Modificar `setup.py` para que tenga tu carrera y padrón
- Correr `infra.ipynb` primero, asi se genera el pickle de la base de datos
- Correr cualquiera de los notebooks de análisis
