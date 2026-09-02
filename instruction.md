# Anleitung
1. Folder speichern, wo du optimalerweise 100gb+ frei hast
2. `bash ./download_at_dtm.sh` um das 5M DGM GeoTiff herunterzuladen und umzuprojizieren (setzt $GEOTIFF_FILE automatisch für den dritten schritt)
3. `bash ./create_tiles.sh` generiert die einzelnen files für die 256x256 tiles
4. `poetry env use <path-to-python3.13>` Poetry environment, sollte hoffentlich alle dependencies haben
5. `poetry install` dependency-Installation
6. `python sendable.py`
