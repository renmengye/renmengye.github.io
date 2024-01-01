rm -rf arxiv.tar.gz
tar -czvf arxiv.tar.gz \
figures/*.pdf \
figures/*/*.pdf \
sections/*.tex \
tables/*.tex \
neurips_2020.sty \
main.bbl \
main.tex
