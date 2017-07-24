tests:
	#python3 -m unittest tests.test_stacking.TestStackingLogit
	#python3 -m unittest tests.test_stacking.TestStackingBayesianAverage
	#python3 -m unittest tests.test_stacking.TestStackingBayesianAverageMCMC
	#python3 -m unittest tests.test_stacking.TestStackingMLR
	#python3 -m unittest tests.test_stacking.TestAgnosticBayesian
	#python3 -m unittest tests.test_stacking.TestStackingMLR
	#python3 -m unittest tests.test_stacking.TestStackingLogit
	python3 -m unittest tests.test_ensemble_selection.TestEnsembleSelectionStandard
	#python3 -m unittest tests.test_ensemble_selection.TestEnsembleSelectionSortedInit
	python3 -m unittest tests.test_ensemble_selection.TestEnsembleSelectionWithReplacements
	#python3 -m unittest tests.test_ensemble_selection.TestEnsembleSelectionSortedInit
	#python3 -m unittest tests.test_wrappers.TestPruning
	#python3 -m unittest tests.test_wrappers.TestBagging
	rm -rf *.pyc __pycache__
	rm -rf */*.pyc */__pycache__
test: tests

clean:
	rm -rf *.pyc __pycache__
	rm -rf */*.pyc */__pycache__
	rm -rf */*/*.pyc */__pycache__

.PHONY: tests test
