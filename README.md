# FIUBA-nx

Análisis de los datos del FIUBA-Map.

---

**¿Qué es esto?**

El [FIUBA-Map](https://fede.dm/FIUBA-Map/) es una herramienta de FIUBA para que los alumnos puedan ir marcando su carrera facultativa (qué materias cursaron, qué nota se sacaron, etc).

Si bien no es una red social (en el sentido criollo de "red social"), no deja de ser una herramienta que contiene información de personas y de cómo estas personas se relacionan entre sí. Estos datos no son directos: en el FMap en ningún momento marcás tu relación con otros alumnos (no existe botón de "Yo hice un trabajo práctico con X"), pero sí pueden empezar a inferirse (vos y yo hacemos la misma carrera, vos y ella cursaron la misma materia, etc). ¡De eso se trata este trabajo práctico para la materia de redes sociales! Encontrar y descubrir las relaciones implícitas entre distintos alumnos de la misma carrera.

El trabajo es más que nada un análisis exploratorio de qué datos tenemos y cómo se relacionan entre sí, dónde podemos ir utilizando conceptos teóricos de redes sociales, y concluye en poder responder:

- ¿Podemos decir que el Fiuba Map es una red social?
- ¿Con qué otros alumnos puedo hacer TPs en un futuro?
- ¿Qué materias electivas me conviene cursar?
- ¿Cuáles son mis notas menos confiables?

---

**¿Cómo funciona?**

El trabajo se divide en la infra y los grafos:

- `infra.ipynb`: donde se levantan los datos del FMap, se parsean y se convierten en un dataframe de Pandas que pueda ser leído por NetworkX, para poder armar un grafo en base a eso.
- `{grafazo,graphito,grafon}.ipynb`: para cada una de las preguntas planteadas queremos ver qué datos tenemos que puedan responderlas, qué grafo podemos armar que nos ayude a contestarlas, y efectivamente lograr contestarlas.

---

**¿Cómo lo corro localmente para poder ver qué electivas me convienen, mis compañeros de TP, y demás?**

Primero, instalar Poetry y setupear el repo.

```zsh
# (En algún virtualenv)
poetry install
poetry run pre-commit install
poetry run jupyter lab
```

Ahora sí:

- Modificar `config.py` para que tenga tu carrera y padrón
- Correr `infra.ipynb` primero, así se genera el pickle de la base de datos
- Correr cualquiera de los notebooks de análisis
