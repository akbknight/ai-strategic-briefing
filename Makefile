.PHONY: run run-sample test lint clean

run:
	python briefing.py --input $(INPUT) --title "$(TITLE)"

run-sample:
	python briefing.py --input data/raw/sample_retail_sales.csv --title "US Retail Sample"

test:
	python -m pytest tests/ -v

lint:
	python -m py_compile src/**/*.py && echo "Syntax OK"

clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null; true
	rm -f report_*.html
