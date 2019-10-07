package moa.classifiers.multilabel;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.yahoo.labs.samoa.instances.*;
import moa.classifiers.AbstractMultiLabelLearner;
import moa.classifiers.MultiLabelClassifier;
import moa.core.Measurement;

import java.util.*;

/**
* Multi-label Punitive kNN with Self-Adjusting Memory for Drifting Data Streams
* 
* @author Alberto Cano
*/

public class MLSAMPkNN extends AbstractMultiLabelLearner implements MultiLabelClassifier {

	private static final long serialVersionUID = 1L;

	public IntOption kOption = new IntOption("k", 'k', "The number of neighbors", 3, 1, Integer.MAX_VALUE);

	public IntOption maxWindowSize = new IntOption("maxWindowSize", 'w', "The maximum number of instances to store", 1000, 1, Integer.MAX_VALUE);

	public IntOption minWindowSize = new IntOption("minWindowSize", 'm', "The minimum number of instances to sotre",   50, 1, Integer.MAX_VALUE);

	public FloatOption penalty = new FloatOption("penalty", 'p', "Penalty ratio", 1, 0, Float.MAX_VALUE);

	public FloatOption reductionRatio = new FloatOption("reductionRatio", 'r', "Reduction ratio", 0.5, 0, 1);

	private String[] metrics = {"Subset Accuracy", "Hamming Score"};

	public MultiChoiceOption metric = new MultiChoiceOption("metric", 'e', "Choose metric used to adjust memory", metrics, metrics, 0);

	private int numLabels;
	private List<Instance> window;
	private double[][] distanceMatrix;
	private double[] attributeRangeMin;
	private double[] attributeRangeMax;
	private Map<Integer, List<Integer>> predictionHistories;
	private Map<Instance, Integer> errors;

	@Override
	public void setModelContext(InstancesHeader context) {
		try {
			numLabels = context.numOutputAttributes();
			window = new ArrayList<Instance>();
			attributeRangeMin = new double[context.numInputAttributes()];
			attributeRangeMax = new double[context.numInputAttributes()];
			distanceMatrix = new double[maxWindowSize.getValue()][maxWindowSize.getValue()];
			predictionHistories = new HashMap<Integer, List<Integer>>();
			errors = new HashMap<Instance, Integer>();

		} catch(Exception e) {
			System.err.println("Error: no Model Context available.");
			e.printStackTrace();
			System.exit(1);
		}
	}

	@Override
	public void resetLearningImpl() {
		if(window != null)
		{
			window.clear();
			distanceMatrix = new double[maxWindowSize.getValue()][maxWindowSize.getValue()];
			predictionHistories = new HashMap<Integer, List<Integer>>();
			errors = new HashMap<Instance, Integer>();
		}
	}

	@Override
	public void trainOnInstanceImpl(MultiLabelInstance inst) {

		window.add(inst);

		updateRanges(inst);

		int windowSize = window.size();

		get1ToNDistances(inst, window, distanceMatrix[windowSize-1]);

		List<Instance> discarded = new ArrayList<Instance>();

		for(Map.Entry<Instance, Integer> entry : errors.entrySet())
		{
			if(entry.getValue() > penalty.getValue() * numLabels)
			{
				for(int idx = 0; idx < windowSize; idx++)
				{
					if(window.get(idx) == entry.getKey())
					{
						for (int i = idx; i < windowSize-1; i++)
							for (int j = idx; j < i; j++)
								distanceMatrix[i][j] = distanceMatrix[i+1][j+1];

						discarded.add(window.get(idx));
						window.remove(idx);
						windowSize--;
						break;
					}
				}
			}
		}

		for(Instance instance : discarded)
			errors.remove(instance);

		int newWindowSize = getNewWindowSize();

		if (newWindowSize < windowSize) {
			int diff = windowSize - newWindowSize;

			for (int i = 0; i < diff; i++)
				errors.remove(window.get(i));

			window = window.subList(diff, windowSize);

			for (int i = 0; i < newWindowSize; i++)
				for (int j = 0; j < i; j++)
					distanceMatrix[i][j] = distanceMatrix[diff+i][diff+j];
		}

		if (newWindowSize == maxWindowSize.getValue()) {

			for (int i = 0; i < newWindowSize-1; i++)
				for (int j = 0; j < i; j++)
					distanceMatrix[i][j] = distanceMatrix[i+1][j+1];

			errors.remove(window.get(0));
			window.remove(0);
		}
	}

	/**
	 * Predicts the label of a given sample
	 */
	@Override
	public Prediction getPredictionForInstance(MultiLabelInstance instance) {

		MultiLabelPrediction prediction = new MultiLabelPrediction(instance.numberOutputTargets());

		double distances[] = new double[window.size()];
		get1ToNDistances(instance, window, distances);
		int nnIndices[] = nArgMin(Math.min(distances.length, kOption.getValue()), distances);
		prediction = getPrediction(nnIndices, window);

		for(int nnIdx : nnIndices)
		{
			int error = 0;

			for(int l = 0; l < numLabels; l++)
				if(window.get(nnIdx).classValue(l) != instance.classValue(l))
					error++;

			if(error != 0)
			{
				Integer instanceErrors = errors.remove(window.get(nnIdx));

				if(instanceErrors == null)
					errors.put(window.get(nnIdx), error);
				else
					errors.put(window.get(nnIdx), instanceErrors.intValue() + error);
			}
		}

		return prediction;
	}

	/**
	 * Returns the votes for each label.
	 */
	private MultiLabelPrediction getPrediction(int[] nnIndices, List<Instance> instances) {

		MultiLabelPrediction prediction = new MultiLabelPrediction(numLabels);

		for(int j = 0; j < numLabels; j++)
		{
			int count = 0;

			for (int nnIdx : nnIndices)
				if(instances.get(nnIdx).classValue(j) == 1)
					count++;

			double relativeFrequency = count / (double) nnIndices.length;

			prediction.setVotes(j, new double[]{1.0 - relativeFrequency, relativeFrequency});
		}

		return prediction;
	}

	private Integer getMetricSums(Instance instance, MultiLabelPrediction prediction) {
		int correct = 0;

		/** preset threshold */
		double t = 0.5;

		for (int j = 0; j < prediction.numOutputAttributes(); j++) {
			int yp = (prediction.getVote(j, 1) >= t) ? 1 : 0;
			correct += ((int) instance.classValue(j) == yp) ? 1 : 0;
		}

		return correct;
	}

	private double getMetricFromHistory(List<Integer> history) {

		double metric = 0.0;

		if(this.metric.getChosenLabel() == "Subset Accuracy")
		{
			for(Integer instanceSum : history)
				metric += (instanceSum == numLabels) ? 1 : 0;
		}
		else if (this.metric.getChosenLabel() == "Hamming Score")
		{
			for(Integer instanceSum : history)
				metric += instanceSum / (double) numLabels;
		}

		return metric / history.size();
	}

	/**
	 * Returns the n smallest indices of the smallest values (sorted).
	 */
	private int[] nArgMin(int n, double[] values, int startIdx, int endIdx) {

		int indices[] = new int[n];

		for (int i = 0; i < n; i++){
			double minValue = Double.MAX_VALUE;
			for (int j = startIdx; j < endIdx + 1; j++){

				if (values[j] < minValue){
					boolean alreadyUsed = false;
					for (int k = 0; k < i; k++){
						if (indices[k] == j){
							alreadyUsed = true;
						}
					}
					if (!alreadyUsed){
						indices[i] = j;
						minValue = values[j];
					}
				}
			}
		}
		return indices;
	}

	public int[] nArgMin(int n, double[] values) {
		return nArgMin(n, values, 0, values.length-1);
	}

	/**
	 * Computes the Euclidean distance between one sample and a collection of samples in an 1D-array.
	 */
	private void get1ToNDistances(Instance sample, List<Instance> samples, double[] distances) {

		for (int i = 0; i < samples.size(); i++)
			distances[i] = getDistance(sample, samples.get(i));
	}

	/**
	 * Returns the Euclidean distance.
	 */
	private double getDistance(Instance instance1, Instance instance2) {

		double distance = 0;

		if(instance1.numValues() == instance1.numAttributes()) // Dense Instance
		{
			for(int i = 0; i < instance1.numInputAttributes(); i++)
			{
				double val1 = instance1.valueInputAttribute(i);
				double val2 = instance2.valueInputAttribute(i);

				if(attributeRangeMax[i] - attributeRangeMin[i] != 0)
				{
					val1 = (val1 - attributeRangeMin[i]) / (attributeRangeMax[i] - attributeRangeMin[i]);
					val2 = (val2 - attributeRangeMin[i]) / (attributeRangeMax[i] - attributeRangeMin[i]);
					distance += (val1 - val2) * (val1 - val2);
				}
			}
		}
		else // Sparse Instance
		{
			int firstI = -1, secondI = -1;
			int firstNumValues  = instance1.numValues();
			int secondNumValues = instance2.numValues();
			int numAttributes   = instance1.numAttributes();
			int numOutputs      = instance1.numOutputAttributes();

			for (int p1 = 0, p2 = 0; p1 < firstNumValues || p2 < secondNumValues;) {

				if (p1 >= firstNumValues) {
					firstI = numAttributes;
				} else {
					firstI = instance1.index(p1);
				}

				if (p2 >= secondNumValues) {
					secondI = numAttributes;
				} else {
					secondI = instance2.index(p2);
				}

				if (firstI < numOutputs) {
					p1++;
					continue;
				}

				if (secondI < numOutputs) {
					p2++;
					continue;
				}

				if (firstI == secondI) {
					int idx = firstI - numOutputs;
					if(attributeRangeMax[idx] - attributeRangeMin[idx] != 0)
					{
						double val1 = instance1.valueSparse(p1);
						double val2 = instance2.valueSparse(p2);
						val1 = (val1 - attributeRangeMin[idx]) / (attributeRangeMax[idx] - attributeRangeMin[idx]);
						val2 = (val2 - attributeRangeMin[idx]) / (attributeRangeMax[idx] - attributeRangeMin[idx]);
						distance += (val1 - val2) * (val1 - val2);
					}
					p1++;
					p2++;
				} else if (firstI > secondI) {
					int idx = secondI - numOutputs;
					if(attributeRangeMax[idx] - attributeRangeMin[idx] != 0)
					{
						double val2 = instance2.valueSparse(p2);
						val2 = (val2 - attributeRangeMin[idx]) / (attributeRangeMax[idx] - attributeRangeMin[idx]);
						distance += (val2) * (val2);
					}
					p2++;
				} else {
					int idx = firstI - numOutputs;
					if(attributeRangeMax[idx] - attributeRangeMin[idx] != 0)
					{
						double val1 = instance1.valueSparse(p1);
						val1 = (val1 - attributeRangeMin[idx]) / (attributeRangeMax[idx] - attributeRangeMin[idx]);
						distance += (val1) * (val1);
					}
					p1++;
				}
			}
		}

		return Math.sqrt(distance);
	}

	private void updateRanges(MultiLabelInstance instance) {
		for(int i = 0; i < instance.numInputAttributes(); i++)
		{
			if(instance.valueInputAttribute(i) < attributeRangeMin[i])
				attributeRangeMin[i] = instance.valueInputAttribute(i);
			if(instance.valueInputAttribute(i) > attributeRangeMax[i])
				attributeRangeMax[i] = instance.valueInputAttribute(i);
		}
	}

	/**
	 * Returns the bisected size which maximized the metric
	 */
	private int getNewWindowSize() {

		int numSamples = window.size();
		if (numSamples < 2 * minWindowSize.getValue())
			return numSamples;
		else {
			List<Integer> numSamplesRange = new ArrayList<Integer>();
			numSamplesRange.add(numSamples);
			while (numSamplesRange.get(numSamplesRange.size() - 1) >= 2 * minWindowSize.getValue())
				numSamplesRange.add((int) (numSamplesRange.get(numSamplesRange.size() - 1) * reductionRatio.getValue()));

			Iterator<Integer> it = predictionHistories.keySet().iterator();
			while (it.hasNext()) {
				Integer key = (Integer) it.next();
				if (!numSamplesRange.contains(numSamples - key))
					it.remove();
			}

			List<Double> metricList = new ArrayList<Double>();
			for (Integer numSamplesIt : numSamplesRange) {
				int idx = numSamples - numSamplesIt;
				List<Integer> predHistory;
				if (predictionHistories.containsKey(idx))
					predHistory = getIncrementalTestTrainPredHistory(window, idx, predictionHistories.get(idx));
				else
					predHistory = getTestTrainPredHistory(window, idx);

				predictionHistories.put(idx, predHistory);

				metricList.add(getMetricFromHistory(predHistory));
			}
			int maxMetricIdx = metricList.indexOf(Collections.max(metricList));
			int windowSize = numSamplesRange.get(maxMetricIdx);

			if (windowSize < numSamples)
				adaptHistories(maxMetricIdx);

			return windowSize;
		}
	}

	/**
	 * Creates a prediction history from the scratch.
	 */
	private List<Integer> getTestTrainPredHistory(List<Instance> instances, int startIdx) {

		List<Integer> predictionHistory = new ArrayList<Integer>();

		for (int i = startIdx; i < instances.size(); i++) {
			int nnIndices[] = nArgMin(Math.min(kOption.getValue(), i - startIdx), distanceMatrix[i], startIdx, i-1);
			MultiLabelPrediction prediction = getPrediction(nnIndices, instances);
			predictionHistory.add(getMetricSums(instances.get(i),prediction));
		}

		return predictionHistory;
	}

	/**
	 * Creates a prediction history incrementally by using the previous predictions.
	 */
	private List<Integer> getIncrementalTestTrainPredHistory(List<Instance> instances, int startIdx, List<Integer> predictionHistory) {

		for (int i = startIdx + predictionHistory.size(); i < instances.size(); i++) {
			int nnIndices[] = nArgMin(Math.min(kOption.getValue(), distanceMatrix[i].length), distanceMatrix[i], startIdx, i-1);
			MultiLabelPrediction prediction = getPrediction(nnIndices, instances);
			predictionHistory.add(getMetricSums(instances.get(i),prediction));
		}

		return predictionHistory;
	}

	/**
	 * Removes predictions of the largest window size and shifts the remaining ones accordingly.
	 */
	private void adaptHistories(int numberOfDeletions) {
		for (int i = 0; i < numberOfDeletions; i++){
			SortedSet<Integer> keys = new TreeSet<Integer>(predictionHistories.keySet());
			predictionHistories.remove(keys.first());
			keys = new TreeSet<Integer>(predictionHistories.keySet());
			for (Integer key : keys){
				List<Integer> predHistory = predictionHistories.remove(key);
				predictionHistories.put(key-keys.first(), predHistory);
			}
		}
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		return null;
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
	}

	public boolean isRandomizable() {
		return false;
	}
}