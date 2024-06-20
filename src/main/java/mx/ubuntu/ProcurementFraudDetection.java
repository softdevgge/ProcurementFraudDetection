package mx.ubuntu;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

public class ProcurementFraudDetection {
 public static void main(String[] args) throws Exception {
        // Sample Data
        List<Contract> contracts = new ArrayList<>();
        contracts.add(new Contract("Vendor A", 1000.0, 30, 8, false));
        contracts.add(new Contract("Vendor B", 5000.0, 60, 3, true));
        contracts.add(new Contract("Vendor C", 1500.0, 90, 5, false));
        contracts.add(new Contract("Vendor D", 2000.0, 120, 2, true));
        contracts.add(new Contract("Vendor E", 3000.0, 45, 7, false));
        contracts.add(new Contract("Vendor A", 10000.0, 30, 8, false));
        contracts.add(new Contract("Vendor A", 50000.0, 60, 3, true));
        contracts.add(new Contract("Vendor B", 15000.0, 90, 5, false));
        contracts.add(new Contract("Vendor B", 20000.0, 120, 2, true));
        contracts.add(new Contract("Vendor C", 30000.0, 45, 7, false));
        // Add more data...

        // Convert data to Weka Instances
        Instances data = createInstances(contracts);

        // Train the model
        Classifier classifier = new J48();
        classifier.buildClassifier(data);

        // Evaluate the model
        Evaluation evaluation = new Evaluation(data);
        evaluation.crossValidateModel(classifier, data, 10, new Random(1));
        System.out.println(evaluation.toSummaryString());

        // Classify a new contract
        double[] values = new double[]{
                data.attribute(0).indexOfValue("Vendor A"),
                60000.0,
                60,
                3
        };
        DenseInstance newInstance = new DenseInstance(1.0, values);
        newInstance.setDataset(data);
        double result = classifier.classifyInstance(newInstance);
        System.out.println("Prediccion (1 = Fraude, 0 = Not Fraude): " + result);
    }

    private static Instances createInstances(List<Contract> contracts) {
        ArrayList<Attribute> attributes = new ArrayList<>();
        ArrayList<String> vendorNames = new ArrayList<>();
        for (Contract contract : contracts) {
            if (!vendorNames.contains(contract.getVendorName())) {
                vendorNames.add(contract.getVendorName());
            }
        }
        attributes.add(new Attribute("vendorName", vendorNames));
        attributes.add(new Attribute("contractValue"));
        attributes.add(new Attribute("contractDuration"));
        attributes.add(new Attribute("pastPerformance"));
        ArrayList<String> classValues = new ArrayList<>();
        classValues.add("false");
        classValues.add("true");
        attributes.add(new Attribute("isFraud", classValues));

        Instances data = new Instances("Contracts", attributes, contracts.size());
        data.setClassIndex(data.numAttributes() - 1);

        for (Contract contract : contracts) {
            double[] values = new double[data.numAttributes()];
            values[0] = data.attribute(0).indexOfValue(contract.getVendorName());
            values[1] = contract.getContractValue();
            values[2] = contract.getContractDuration();
            values[3] = contract.getPastPerformance();
            values[4] = contract.isFraud() ? data.attribute(4).indexOfValue("true") : data.attribute(4).indexOfValue("false");

            data.add(new DenseInstance(1.0, values));
        }

        return data;
    }
}

class Contract {
    private String vendorName;
    private double contractValue;
    private int contractDuration; // in days
    private int pastPerformance; // some performance metric
    private boolean isFraud; // target variable

    public Contract(String vendorName, double contractValue, int contractDuration, int pastPerformance, boolean isFraud) {
        this.vendorName = vendorName;
        this.contractValue = contractValue;
        this.contractDuration = contractDuration;
        this.pastPerformance = pastPerformance;
        this.isFraud = isFraud;
    }

    public String getVendorName() {
        return vendorName;
    }

    public void setVendorName(String vendorName) {
        this.vendorName = vendorName;
    }

    public double getContractValue() {
        return contractValue;
    }

    public void setContractValue(double contractValue) {
        this.contractValue = contractValue;
    }

    public int getContractDuration() {
        return contractDuration;
    }

    public void setContractDuration(int contractDuration) {
        this.contractDuration = contractDuration;
    }

    public int getPastPerformance() {
        return pastPerformance;
    }

    public void setPastPerformance(int pastPerformance) {
        this.pastPerformance = pastPerformance;
    }

    public boolean isFraud() {
        return isFraud;
    }

    public void setFraud(boolean isFraud) {
        this.isFraud = isFraud;
    }


    
}

