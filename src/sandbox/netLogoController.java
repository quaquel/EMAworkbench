import org.nlogo.app.App;
import java.awt.EventQueue;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.util.Random;


public class Controller 
{
	public static void main(String[] argv) 
	{
		Controller controller = new Controller();
		int runLength=100;
		App.main(argv);
	    try 
	    {
	    	EventQueue.invokeAndWait
	        ( 	new Runnable()
	        	{ 	public void run() 
	        		{
	        			try 
	        			{
	        				App.app.open ("lib/InfoDiffNetwork-FullConnected.nlogo");  //Path to the netlogo model to be controlled
	                    }
	        			catch( java.io.IOException ex ) 
	        			{
	        				ex.printStackTrace();
	                    }
	                 } 
	        	} 
	        );
	        
	    	//Pass the parameter values by calling the command function. The argument of the function is a string in the following form "set [parameter Name] [parameter value]"
    		App.app.command("set util "+1.2); //sets the parameter named "util" to 1.2
	    	App.app.command("set percUtil-potential-avg "+5);
	    	App.app.command("set percUtil-adopted-avg "+6);
	    	//Model is prepared for simulation (i.e. initialized) by passing the "setup" command
	    	App.app.command("setup");
	    	
	    	//You have to invoke "Go" command at every time step, if you wish to record the data from each time point
	    	//It sounds stupidly primitive, so I will double check this. At least in the version, I used 3 years ago, it worked like this
	    	for(int i=0;i<runLength;i++)
	        {
	    		//Create a data object for a variable of interest
	    		Object data= (App.app.report("countAdop"));
	        	double x = ((Double)data).doubleValue();
	        	System.out.println(x);
	        	
	        	//The command to simulate the model one time step
	        	App.app.command("go");
	        }	        
	    }
	    catch(Exception ex) 
	    {
	    	ex.printStackTrace();
	    }   
	}
}


	



