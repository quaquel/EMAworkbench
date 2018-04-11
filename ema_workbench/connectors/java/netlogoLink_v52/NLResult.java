package netlogoLink_v52;

import java.util.LinkedHashMap;
import java.text.ParseException;
import org.nlogo.api.LogoList;

public class NLResult {

	private String type = null;
	private Object resultValue = null;
	private Integer NumberNestedLists = null;
	private String[] NestedTypes = null;
	
	public void setResultValue(Object o) throws Exception {
		logoToType(o);
	}
	
	public String getType() {
		return type;
	}

	public String getResultAsString() {
		return ((String)resultValue).toString();
	}
	
	public double getResultAsDouble() {
		return ((Double)resultValue).doubleValue();
	}
	
	public boolean getResultAsBoolean() {
		return ((Boolean)resultValue).booleanValue();
	}
	
	public Integer getResultAsInteger() {
		return (Integer)resultValue;
	}
	
	public double[] getResultAsDoubleArray() {
		return (double[])resultValue;
	}

	public int[] getResultAsIntegerArray() {
		return (int[])resultValue;
	}	
	
	public boolean[] getResultAsBooleanArray() {
		return (boolean[])resultValue;
	}

	public String[] getResultAsStringArray() {
		return (String[])resultValue;
	}	
	
	public Object getResultAsObject() {
		return resultValue;
	}
	
	public Object[] getResultAsObjectArray() {
		return (Object[])resultValue;
	}
	
	
	
	private void logoToType( Object o ) throws Exception {
		if(o instanceof LogoList)
		{
			type = "LogoList";
			org.nlogo.api.LogoList loli = (org.nlogo.api.LogoList)o;
			resultValue = cast_logolist(loli, false);
		}
		else if (o instanceof String) {
			type = "String";
			resultValue = ((String)o).toString();
		}
		else if (o instanceof Integer) {
			type = "Integer";
			resultValue = ((Integer)o).intValue();
		}
		else if (o instanceof Double) {
			type = "Double";
			resultValue = ((Double)o).doubleValue();
		}
		else if (o instanceof Boolean) {
			type = "Boolean";
			resultValue = ((Boolean)o).booleanValue();
		}
		else {
			type = "Unknown";
			resultValue = null;
			throw new Exception("Found unknown datatype: "+o);
		}
	}
	
	
	/**
	 * Method to transform a NetLogo List and put via rni.
	 * @param obj instance of LogoList
	 * @return long containing rni reference value
	 */	
	private Object cast_logolist(LogoList logolist, Boolean recursive) throws Exception
	{
		try
		{
			
    		if (logolist.get(0) instanceof LogoList)
    		{ 
    			Object[] lilist = new Object[logolist.size()];
    			NestedTypes = new String[logolist.size()];
				for (int i=0; i<logolist.size(); i++)
				{
					NLResult nestedResult = new NLResult();
					nestedResult.setResultValue(logolist.get(i));
					lilist[i] = nestedResult; //cast_logolist((LogoList)logolist.get(i), true);
				}
    			type = "NestedList";
				return lilist;
    		}

    	    if (logolist.get(0) instanceof java.lang.String)
    	    {
           		String[] stringlist = new String[logolist.size()];
				for (int i=0; i<logolist.size(); i++)
				{
					stringlist[i] = (String)logolist.get(i);
				}
				if (!recursive)
					type = "StringList";
				return stringlist;
    	    }		    	    

    	    if (logolist.get(0) instanceof java.lang.Double)
    	    {
				double[] dblist = new double[logolist.size()];
				//Double[] dblist = new Double[logolist.size()];
				for (int i=0; i<logolist.size(); i++)
				{
					dblist[i] = ((java.lang.Double)logolist.get(i)).doubleValue();
					//dblist[i] = (java.lang.Double)logolist.get(i);
				}     	
				if (!recursive)
					type = "DoubleList";
				return dblist; 	
    	   }   		    	   

    	   if (logolist.get(0) instanceof java.lang.Boolean)
    	   {
    	       	//int[] intbool= new int[logolist.size()];
    	       	boolean[] boollist = new boolean[logolist.size()];
				for (int i=0; i<logolist.size(); i++)
				{
					//if ((Boolean)logolist.get(i))
    	       		//	intbool[i] = 1;
    	       		//else
    	       		//	intbool[i] = 0;
					if (!recursive)
						type = "BoolList";
					boollist[i] = ((java.lang.Boolean)logolist.get(i)).booleanValue();
				}
				//invalue = intbool; 	
				return boollist;
    	   }
    	   
    	   if (logolist.get(0) instanceof java.lang.Integer)
    	   {
    	       	int[] intlist = new int[logolist.size()];
				for (int i=0; i<logolist.size(); i++)
				{
					if (!recursive)
						type = "IntegerList";
					intlist[i] = ((java.lang.Integer)logolist.get(i)).intValue();
				}
				//invalue = intbool; 	
				return intlist;
    	   }
		}
		catch (Exception ex)
		{
			//System.out.println("Error in putRNI: "+ex);
			throw new ParseException("Java error in converting result: "+ex, 1);
		}
		return null;
	}
	
	
}
