LATEXMK := /Library/TeX/texbin/latexmk

.PHONY: all clean

all:
	@mkdir -p build
	$(LATEXMK) -xelatex -interaction=nonstopmode -halt-on-error -outdir=build main.tex

clean:
	$(LATEXMK) -C -outdir=build main.tex
