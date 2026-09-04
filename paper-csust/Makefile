LATEXMK ?= /Library/TeX/texbin/latexmk

.PHONY: thesis clean

thesis:
	$(LATEXMK) -xelatex -interaction=nonstopmode -halt-on-error -outdir=build thesis.tex

clean:
	$(LATEXMK) -C -outdir=build thesis.tex
