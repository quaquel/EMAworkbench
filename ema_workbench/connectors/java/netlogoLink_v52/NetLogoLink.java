package netlogoLink_v52;


// partly based on the work of Uri Wilensky's Mathematica link:
//(C) 2007 Uri Wilensky. This code may be freely copied, distributed,
//altered, or otherwise used by anyone for any legal purpose.


import org.nlogo.headless.HeadlessWorkspace;
import org.nlogo.api.CompilerException;
import org.nlogo.api.LogoException;
import org.nlogo.app.App;
import java.awt.EventQueue;
import java.awt.Frame;
import java.security.Permission;
import javax.swing.JOptionPane;

import java.lang.Thread;


public class NetLogoLink {
	private org.nlogo.workspace.Controllable workspace = null;
	private java.io.IOException caughtEx = null;
	private boolean isGUIworkspace;
	private static boolean blockExit = true;

	static final SecurityManager securityManager1 = new SecurityManager()
	{
		public void checkPermission(Permission permission)
		{
			if (blockExit) {
				//This Prevents the shutting down of JVM.(in case of System.exit())
				if ("exitVM".equals(permission.getName()))
				{
					JOptionPane.showMessageDialog(null, "system.exit attemted and blocked.", "Error", JOptionPane.OK_CANCEL_OPTION);
					throw new SecurityException("System.exit attempted and blocked.");
				}
			}
		}
	     public void checkExit(int status) {
	    	 if (blockExit) {
				JOptionPane.showMessageDialog(null, "Please use NLQuit() for closing the window.", "Error", JOptionPane.OK_CANCEL_OPTION);
	    	 	//System.out.println("Thread is requesting exit permissions, will be denied");
	            throw new SecurityException("Preventing sub-tool from calling System.exit(" + Integer.toString(status) + ")!");
	    	 }
	    }
	};
	
	public NetLogoLink(Boolean isGUImode, Boolean is3d)
	{
		/**
		 * Instantiates a link to netlogo
		 * 
		 * @param isGuiMode	boolean indicated whether netlogo should be run
		 * 		  			with gui or in headless mode
		 * @param is3d      boolean indicated whether to run netlogo in 2d or 
		 * 				   	in 3d mode.
		 * 
		 */
		
		try
		{
			System.setSecurityManager(securityManager1);
			System.setProperty("org.nlogo.is3d", is3d.toString());
			isGUIworkspace = isGUImode.booleanValue();
			if( isGUIworkspace ) {
				App.main( new String[] { } ) ;
				workspace = App.app();
				org.nlogo.util.Exceptions.setHandler
					( new org.nlogo.util.Exceptions.Handler() {
							public void handle( Throwable t ) {
								throw new RuntimeException(t.getMessage());
							} } );
			}
			else
				workspace = HeadlessWorkspace.newInstance() ;
		}
		catch (Exception ex) {
			JOptionPane.showMessageDialog(null, "Error in Constructor NLink:"+ex, "Error", JOptionPane.OK_CANCEL_OPTION);			
		}
	}	
	
	public void killWorkspace()
	{
		/**
		 * 	it is not possible to close NetLogo by its own closing method, 
		 *  because it is based on System.exit(0) which will result in a 
		 *  termination of the JVM, jpype and finally python. Therefore, we 
		 *  only dispose the thread, we can find. This is the identical to
		 *  how it is done in RNetLogo.
		 */
		
		try
		{
			blockExit = false;

			if (isGUIworkspace) {
				for (int i=0; i<((App)workspace).frame().getFrames().length; i++) {
					java.awt.Frame frame = ((App)workspace).frame().getFrames()[i];
					
					frame.dispose();
				}
				Thread.currentThread().interrupt();
			}
			else {
				((HeadlessWorkspace)workspace).dispose();
			}
		}
		catch (Exception ex) {
			JOptionPane.showMessageDialog(null, "Error in killing workspace:"+ex, "Error", JOptionPane.OK_CANCEL_OPTION);
		}
		workspace = null;
		System.gc();
	}


	public void loadModel(final String path)
		throws java.io.IOException, LogoException, CompilerException, InterruptedException
	{
		/**
		 * load a model
		 * 
		 * @param path	a string with the absolute path of the model
		 * @throws IOException, LogoException, CompilerException, InterruptedException
		 * 
		 */
			caughtEx = null;
			if ( isGUIworkspace ) {
				try {
					EventQueue.invokeAndWait ( 
						new Runnable() {
							public void run() {
								try
								{ App.app().open(path); }
								catch( java.io.IOException ex)
								{ caughtEx = ex; }
							} } );
				}
				catch( java.lang.reflect.InvocationTargetException ex ) {
					JOptionPane.showMessageDialog(null, "Error in loading model:"+ex, "Error", JOptionPane.OK_CANCEL_OPTION);
					throw new RuntimeException(ex.getMessage());
				}
				if( caughtEx != null ) {
					throw caughtEx;
				}
			}
			else {
				try {
					if (workspace != null)
						((HeadlessWorkspace)workspace).dispose();
					workspace = HeadlessWorkspace.newInstance() ;
					workspace.open(path);
				}
				catch( java.io.IOException ex) {
					JOptionPane.showMessageDialog(null, "Error in loading model:"+ex, "Error", JOptionPane.OK_CANCEL_OPTION);

					if (workspace != null)
						((HeadlessWorkspace)workspace).dispose();
					workspace = HeadlessWorkspace.newInstance() ;
					throw ex;
				}
			}
	}

	public void command(final String s)
		throws LogoException, CompilerException
	{
		/**
		 * execute the supplied command in netlogo. This method
		 * is a wrapper around netlogo's command.
		 * 
		 * @param s	a valid netlogo command
		 * @throws LogoException, ComplierException
		 * 
		 */
		
		workspace.command(s);
	}

	/* returns the value of a reporter.  if it is a LogoList, it will be
	recursively converted to an array of Objects */
	public Object report(String s)
		throws LogoException, CompilerException, Exception
	{
		/** 
		 * Every reporter (commands which return a value) that can be called 
		 * in the NetLogo Command Center can be called with this method. 
		 * Like command, it is essentially a wrapper around netlogo's report
		 * 
		 * @param s	a valid netlogo reporter
		 * 
		 */		
		
		NLResult result = new NLResult();
		result.setResultValue(workspace.report(s));
		return result;
	}
	
}

 