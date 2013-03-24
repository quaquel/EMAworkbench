
public class WillemDistance {

	double crisisThold = 0.20;
	double trendThold = 0.001;
	
	double wIfCrisis = 10000;
	double wNoOfCrisis = 1;
	double wTrend = 1000;
			
	public double distance(double[] ds1, double[] ds2) {
		double[][] features = new double[2][3];
		features[0] = constructFeatureVector(ds1);
		features[1] = constructFeatureVector(ds2);
				
		double comp1 = Math.abs(features[0][0] - features[1][0]);
		double comp2 = Math.abs(features[0][1] - features[1][1]);
		double comp3 = Math.abs(features[0][2] - features[1][2]);
		double distance = (comp1*wIfCrisis)+(comp2*wNoOfCrisis)+(comp3*wTrend);
			
		return distance;
	}
	
	private int getTrend(double[] ds) {
		double[] slope = new double[ds.length-1];
		double slopeMean 	= 0; 
		double dsMean	= 0;
		int trend = 0;
		
		for(int j=0;j<slope.length;j++){
				slope[j]=ds[j+1]-ds[j];
				slopeMean+= slope[j]/slope.length;
				dsMean+= ds[j]/ds.length;
		}
		
		if(slopeMean/dsMean>trendThold){
			trend = 1;
		}
		else if (slopeMean/dsMean<-1*trendThold){
			trend = -1;
		}
		else trend = 0;
		return trend;
	}

	private int getNoOfCrisis(double[] ds) {
		double[] slope 			= new double[ds.length-1];
		double slopeMean 	= 0; 
		double dsMean	= 0;
		int crisisCount =0;
		
		for(int j=0;j<slope.length;j++){
				slope[j]=ds[j+1]-ds[j];
				slopeMean+= slope[j]/slope.length;
				dsMean+= ds[j]/ds.length;
		}
			
		for(int j=0;j<slope.length;j++){
			//System.out.println("Slope "+j+"  :"+slope[j]+"  level "+ds[j]+"  ratio: "+(Math.abs(slope[j])/ds[j]));
			if(Math.abs(slope[j])/ds[j]>crisisThold){
				//System.out.println("Crisis "+j);
				crisisCount++;
			}
		}
		return crisisCount;
	}

	public double[] constructFeatureVector(double[] ds) {
		double[] features = new double[3];
		features[1] = getNoOfCrisis(ds);
		features[2] = getTrend(ds);
		if(features[1] >0){
			features[0] = 1;
		}
		else features[0] = 0; 
		return features;
	}

	public double[] getFeatureVector(double[] ds){
		double[] features = constructFeatureVector(ds);
		return features;
	}
}
