package edu.vcu.acano.MLSAMPkNN;

import com.yahoo.labs.samoa.instances.MultiLabelPrediction;

import moa.classifiers.MultiLabelClassifier;
import moa.core.InstanceExample;
import moa.core.TimingUtils;
import moa.evaluation.PrequentialMultiLabelPerformanceEvaluator;
import moa.streams.generators.multilabel.MetaMultilabelGenerator;

public class MLSAMPkNN {

	public void run(int maximumNumberInstances)
	{
		// A. Select input for the program (A.1 Data stream generator) or (A.2 Data stream from dataset file)
		
		// Generator as a stream
		MetaMultilabelGenerator stream = new MetaMultilabelGenerator();
		stream.numLabelsOption.setValue(10);
				
		stream.prepareForUse();

		// B. Setup multi-label classifier

		MultiLabelClassifier learner = new moa.classifiers.multilabel.MLSAMPkNN();
		
		learner.setModelContext(stream.getHeader());
		learner.prepareForUse();

		int numberInstances = 0;
		
		PrequentialMultiLabelPerformanceEvaluator evaluator = new PrequentialMultiLabelPerformanceEvaluator();

		long evaluateStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
		
		while (stream.hasMoreInstances() && numberInstances < maximumNumberInstances)
		{
			InstanceExample instance = stream.nextInstance();
			
			MultiLabelPrediction prediction = (MultiLabelPrediction) learner.getPredictionForInstance(instance);
			
			evaluator.addResult(instance, prediction);
			
			learner.trainOnInstance(instance);

			numberInstances++;
		}

		double time = TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread()- evaluateStartTime);

		System.out.println(numberInstances + " instances processed in "+time+" seconds.");
		
		for(int i = 0; i < evaluator.getPerformanceMeasurements().length; i++)
			System.out.println(evaluator.getPerformanceMeasurements()[i].getName() + "\t" + evaluator.getPerformanceMeasurements()[i].getValue());
	}

	public static void main(String[] args) throws Exception
	{
		MLSAMPkNN exp = new MLSAMPkNN();
		exp.run(100);
	}
}