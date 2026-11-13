#!/usr/bin/env python3
"""Crée un GIF animé à partir de tous les .png d'un dossier.

L'ordre des frames suit l'ordre alphanumérique (naturel) des noms de fichiers :
frame_2.png vient avant frame_10.png.

Usage :
    python make_gif.py /chemin/vers/dossier
    python make_gif.py /chemin/vers/dossier -o sortie.gif --duration 100 --loop 0

Dépendance : Pillow  (pip install pillow)
"""

import argparse
import re
import sys
from pathlib import Path

from PIL import Image


def natural_key(path: Path):
    """Clé de tri alphanumérique : 'img2' < 'img10'."""
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", path.name)
    ]


def find_pngs(directory: Path):
    return sorted(
        (p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".png"),
        key=natural_key,
    )


def make_gif(directory: Path, output: Path, duration: int, loop: int) -> None:
    pngs = find_pngs(directory)
    if not pngs:
        sys.exit(f"Aucun fichier .png trouvé dans {directory}")

    print(f"{len(pngs)} fichier(s) .png trouvé(s) :")
    for p in pngs:
        print(f"  {p.name}")

    # Taille de référence = première frame ; les autres sont redimensionnées si besoin
    frames = []
    size = None
    for p in pngs:
        with Image.open(p) as im:
            im = im.convert("RGBA")
            if size is None:
                size = im.size
            elif im.size != size:
                print(f"  [warn] {p.name} : {im.size} -> redimensionné en {size}")
                im = im.resize(size, Image.LANCZOS)
            # Compose sur fond blanc pour éviter les artefacts de transparence
            background = Image.new("RGBA", size, (255, 255, 255, 255))
            background.alpha_composite(im)
            frames.append(background.convert("RGB"))

    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=duration,  # ms par frame
        loop=loop,  # 0 = boucle infinie
        optimize=False,
    )
    print(f"GIF écrit : {output}")


def main():
    parser = argparse.ArgumentParser(description="Crée un GIF à partir des .png d'un dossier.")
    parser.add_argument("directory", type=Path, help="Dossier contenant les .png")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Fichier GIF de sortie (défaut : <directory>/output.gif)",
    )
    parser.add_argument(
        "-d", "--duration", type=int, default=100,
        help="Durée de chaque frame en ms (défaut : 100)",
    )
    parser.add_argument(
        "-l", "--loop", type=int, default=0,
        help="Nombre de boucles, 0 = infini (défaut : 0)",
    )
    args = parser.parse_args()

    if not args.directory.is_dir():
        sys.exit(f"{args.directory} n'est pas un dossier valide")

    output = args.output or args.directory / "output.gif"
    make_gif(args.directory, output, args.duration, args.loop)


if __name__ == "__main__":
    main()