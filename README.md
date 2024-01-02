Repository structure should be as follows:

my_project/
│
├── .git/            # Carpeta del sistema de control de versiones
│
├── venv/            # Carpeta del entorno virtual (agregada a .gitignore)
│
├── .gitignore       # Archivo para especificar archivos/directorios ignorados por Git
│
├── requirements.txt  # Archivo con las dependencias del proyecto
│
├── src/             # Carpeta con el código fuente del proyecto
│   ├── my_module/   # Módulos del proyecto
│   │   ├── __init__.py
│   │   ├── module1.py
│   │   └── module2.py
│   │
│   └── main.py      # Script principal
│
├── tests/           # Carpeta con pruebas unitarias
│   ├── test_module1.py
│   └── test_module2.py
│
├── data/            # Carpeta para datos del proyecto (opcional)
│
├── docs/            # Carpeta para documentación (opcional)
│
├── .vscode/         # Carpeta de configuración de Visual Studio Code (opcional)
│
├── README.md        # Archivo de README para describir el proyecto
│
└── LICENSE          # Archivo de licencia del proyecto

