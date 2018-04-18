/*
 medinology Android 1.0
 */
package com.kyunggi.medinology;

import android.app.*;
import android.content.*;
import android.os.*;
import android.view.*;
import android.view.View.*;
import android.widget.*;
import java.io.*;
import java.util.*;
import java.nio.*;
import android.graphics.*;


public class MainActivity extends Activity implements OnClickListener, CheckBox.OnCheckedChangeListener
{

	//XML 파일 오류등으로 직접 생성할 컴포넌트들
	LinearLayout mainLayout;
	FrameLayout firstFrame;
	ScrollView scrollView;
	LinearLayout inlinearLayout;
	ArrayList<FrameLayout> symptomTabFrames;
	Button BT_gosym;
	Button BT_getResult;
	ArrayList<Button> nextButtons=null;
	ArrayList<Button> backButtons=null;
	CheckBox CB_Preg;
	NumberPicker NP_Age;
	RadioGroup RG_Gender;
	RadioButton RB_Male,RB_Female;
	//native 메소드에 넘겨줄 변수들
	boolean male;
	boolean preg;
	int age;
	ArrayList<Byte> symptombytes=new ArrayList<Byte>();		//원래는 boolean 배열이었으나 호환성을 위해 byte로

	
	boolean bMuJom=false;
	boolean bChijil=false;
    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState)
    {

        super.onCreate(savedInstanceState);
		try
		{
			mainLayout = new LinearLayout(this);
			firstFrame = new FrameLayout(this);
			scrollView = new ScrollView(this);
			inlinearLayout = new LinearLayout(this);
			TextView tvtitle=new TextView(this);
			RG_Gender = new RadioGroup(this);
			RB_Male = new RadioButton(this);
			RB_Female = new RadioButton(this);
			CB_Preg = new CheckBox(this);
			TextView tvage=new TextView(this);
			NP_Age = new NumberPicker(this);
			BT_gosym = new Button(this);
			LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(
				LinearLayout.LayoutParams.MATCH_PARENT,
				LinearLayout.LayoutParams.WRAP_CONTENT
			);
			tvtitle.setText("Medinology - 기초 정보");
			tvtitle.setTextSize(60);
			inlinearLayout.setOrientation(LinearLayout.VERTICAL);
			RG_Gender.setOrientation(RG_Gender.HORIZONTAL);
			RB_Female.setText("여자");
			RB_Male.setText("남자");
			RB_Male.setChecked(false);
			RB_Female.setChecked(false);
			RG_Gender.addView(RB_Male);
			RG_Gender.addView(RB_Female);
			CB_Preg.setText("임신함");
			CB_Preg.setChecked(false);
			CB_Preg.setOnCheckedChangeListener(this);
			RB_Male.setOnCheckedChangeListener(this);
			tvage.setText("나이");
			tvage.setTextSize(30);
			BT_gosym.setText("증상 고르러 가기");
			BT_gosym.setTextSize(40);
			BT_gosym.setOnClickListener(this);
			NP_Age.setMinValue(0);
			NP_Age.setMaxValue(250);
			NP_Age.setWrapSelectorWheel(false);
			NP_Age.setOnLongPressUpdateInterval(100);
			inlinearLayout.addView(tvtitle, p);
			inlinearLayout.addView(RG_Gender, p);
			inlinearLayout.addView(CB_Preg, p);
			inlinearLayout.addView(tvage, p);
			inlinearLayout.addView(NP_Age, p);
			inlinearLayout.addView(BT_gosym, p);
			scrollView.addView(inlinearLayout);
			firstFrame.addView(scrollView);
			mainLayout.addView(firstFrame);
			setContentView(mainLayout);
			LoadSymptoms();
			LoadDiseaseAndMedinames();
			ReadDisease_Drug(diseaseNames.size(), mediNames.size());
			ReadDrug_Detail(mediNames.size(), 4);
			//initWeights("");
			Toast.makeText(this, "symptom =" + new Integer(symptomNames.size()).toString() +
						   "diseasenames" + new Integer(diseaseNames.size()).toString() +
						   "medinames" + new Integer(mediNames.size()).toString() +
						   "drugtablerows" + new Integer(disdrugTable.size()).toString() +
						   "dittablerows" + new Integer(drugditTable.size()).toString() +
						   "drugtablecols" + new Integer(disdrugTable.get(0).size()).toString() +
						   "drugditcols" + new Integer(drugditTable.get(0).size()).toString(), 5);//.show();
		}
		catch (Exception e)
		{
			ByteArrayOutputStream out = new ByteArrayOutputStream();
			PrintStream pinrtStream = new PrintStream(out);
			//e.printStackTrace()하면 System.out에 찍는데,
			// 출력할 PrintStream을 생성해서 건네 준다
			e.printStackTrace(pinrtStream);
			String stackTraceString = out.toString(); // 찍은 값을 가져오고.
			Toast.makeText(this, stackTraceString, 50).show();//보여 준다
		} 	
		//setContentView(R.layout.main);
		//mainLayout = (LinearLayout)findViewById(R.id.mainLayout);
		//firstFrame=(FrameLayout) findViewById(R.id.firstFrame);
		//NP_Age=(NumberPicker) findViewById(R.id.NP_Age);
		//RB_Gender=(RadioButton) findViewById(R.id.RB_Male);
		//CB_Preg= (CheckBox) findViewById(R.id.CB_Preg);
		//BT_gosym = (Button)findViewById(R.id.BtnGoSym);
    }

	@Override
	protected void onDestroy()
	{
		// TODO: Implement this method
		super.onDestroy();
		finalizeNative();
	}
	ArrayList<ArrayList<String>> symptomNames;
	ArrayList<String> kinds;
	ArrayList<CheckBox> cboxes;
	//설정 파일을 읽어 동적으로 페이지를 생성하는 루틴
	private void LoadSymptoms() throws Exception
	{
		//File file  =  new File("Symptoms.txt");		//Root 디렉터리에 접근

		int i=0;
		symptomNames = new ArrayList<ArrayList<String>>();

		kinds = new ArrayList<String>();

		nextButtons = new ArrayList<Button>();
		try
		{
			BufferedReader br  =  new BufferedReader(new InputStreamReader(getResources().openRawResource(R.raw.symptoms), "euc-kr"));

			String line;
			String kind,sym;
			ArrayList<String> buf=null;
			while ((line = br.readLine()) != null)
			{
				buf = new ArrayList<String>();
				if (i % 2 == 0)
				{
					kind = line;
					kinds.add(kind);
					if (i != 0)
					{
						symptomNames.add(new ArrayList<String>(buf));
						//buf.clear();
					}
				}
				else
				{
					line.replaceAll("  ", " ");
					StringTokenizer tokenizer=new StringTokenizer(line, " ");
					while (tokenizer.hasMoreTokens())
					{
						sym = tokenizer.nextToken(" ");	
						buf.add(sym);
					}
				}
				++i;
			}
			if (buf != null)
			{
				symptomNames.add(new ArrayList<String>(buf));
			}

			//buf.clear();

		}
		catch (IOException e)
		{
			Toast.makeText(this, e.toString() + System.getProperties().toString(), 50).show();
		}
		//이제 뷰 동적 생성
		nextButtons.add(BT_gosym);
		symptomTabFrames = new ArrayList<FrameLayout>();
		symptomTabFrames.add(firstFrame);
		int sz=kinds.size();
		for (int j=0;j < sz;++j)
		{
			String symnam=kinds.get(j);
			TextView tv=new TextView(this);
			tv.setText(symnam);
			tv.setTextSize(20);
			FrameLayout frame=new FrameLayout(this);
			ScrollView scr=new ScrollView(this);
			GridLayout grid=new GridLayout(this);
			grid.setColumnCount(3);
			grid.setOrientation(grid.HORIZONTAL);
			GridLayout.LayoutParams par=new GridLayout.LayoutParams();
			GridLayout.Spec gspec=GridLayout.spec(0, 3);
			par.columnSpec = gspec;
			grid.addView(tv, par);
//			CheckBox cb1=new CheckBox(this);
//			cb1.setEnabled(false);
//			cb1.setVisibility(View.INVISIBLE);
//			grid.addView(cb1);
//			CheckBox cb2=new CheckBox(this);
//			cb2.setEnabled(false);
//			cb2.setVisibility(View.INVISIBLE);
//			grid.addView(cb2);
			int ssz=symptomNames.get(j).size();
			
			Toast.makeText(this, new Integer(ssz).toString(), 1);//.show();
			cboxes=new ArrayList<CheckBox>();
			for (int k=0;k < ssz;++k)
			{
				cboxes.add(new CheckBox(this));
				CheckBox cb=cboxes.get(k);
				cb.setText(symptomNames.get(j).get(k));
				cb.setChecked(false);
				grid.addView(cb);
			}
			Button button=new Button(this);
			if (j == sz - 1)
			{
				button.setText("결과 받기");
			}
			else
			{
				button.setText("다음");
			}
			button.setOnClickListener(this);
			nextButtons.add(button);
			grid.addView(button);
			scr.addView(grid);
			frame.addView(scr);
			frame.setVisibility(frame.GONE);
			mainLayout.addView(frame);
			symptomTabFrames.add(frame);
		}
	}
	//약과 질병명 읽어들임
	ArrayList<String> diseaseNames;
	public static ArrayList<String> mediNames;
	private void LoadDiseaseAndMedinames() throws Exception
	{
		//File file  =  new File("Symptoms.txt");		//Root 디렉터리에 접근

		diseaseNames = new ArrayList<String>();
		try
		{
			BufferedReader br  =  new BufferedReader(new InputStreamReader(getResources().openRawResource(R.raw.diseases), "euc-kr"));

			String line,name;
			if ((line = br.readLine()) != null)
			{
				line.replaceAll("	", " ");
				StringTokenizer tokenizer=new StringTokenizer(line, " ");
				while (tokenizer.hasMoreTokens())
				{
					name = tokenizer.nextToken(" ");	
					diseaseNames.add(name);
				}
			}
		}
		catch (IOException e)
		{
			Toast.makeText(this, e.toString(), 50).show();
		}
		mediNames = new ArrayList<String>();
		try
		{
			BufferedReader br  =  new BufferedReader(new InputStreamReader(getResources().openRawResource(R.raw.drugs), "euc-kr"));

			String line,name;
			if ((line = br.readLine()) != null)
			{
				line.replaceAll("	", " ");
				StringTokenizer tokenizer=new StringTokenizer(line, " ");
				while (tokenizer.hasMoreTokens())
				{
					name = tokenizer.nextToken(" ");	
					mediNames.add(name);
				}
			}
		}
		catch (IOException e)
		{
			Toast.makeText(this, e.toString(), 50).show();
		}
	}

	//Weights를 로드한다.
	private void LoadWeights() throws Exception
	{
		//File file  =  new File("Symptoms.txt");		//Root 디렉터리에 접근
		File file = new File("/sdcard/weight.txt");
		if (file.exists())
		{
			file.delete();
		}


		InputStream inStream = null;
		OutputStream outStream = null; 
		try
		{ 
			inStream = getResources().openRawResource(R.raw.weights);
			outStream = new FileOutputStream(file); 
			byte[] buffer = new byte[1024];
			int length; //copy the file content in bytes
			while ((length = inStream.read(buffer)) > 0)
			{
				outStream.write(buffer, 0, length); 
			} 
			inStream.close(); 
			outStream.close();
			Toast.makeText(this, "DB copy success", 1).show();
		}
		catch (IOException e)
		{ 
			e.printStackTrace();
		} 
		initWeights();
	}

	public int[] mediid1={1};
	public int[] mediid2={0};
	public int[] mediid3={0};
	private void Invoke()
	{
		try
		{
			//Toast.makeText(this,"",1).show();
			//자료를 가공하여 Native로 보낸다.
			byte [] syms= toPrimitives(symptombytes);
			symptombytes.clear();
			//Toast.makeText(this, "init start", 1).show();
			//Thread.sleep(1000);
			initData(male, preg, age, 50, syms, 31);
			LoadWeights();
			//Toast.makeText(this, "initweight succ", 1).show();
			calcData();
			int disid1,disid2,disid3;
			disid1 = 0;
			disid2 = 12;
			int prob1=40;
			int prob2=20;
			int prob3=20;
			
			disid1 = getDisID(0);
			disid2 = getDisID(1);
			disid3 = getDisID(2);
			
			prob1 = getProb(0);
			prob2 = getProb(1);
			prob3 = getProb(2);
			if(bMuJom)
			{
				disid1=30;
				prob1=100;
				bMuJom=false;
			}
			if(bChijil)
			{
				disid2=13;
				prob2=100;
				bChijil=false;
			}
			mediid1 = getDrugID(disid1);
			mediid2 = getDrugID(disid2);
			mediid3 = getDrugID(disid3);
			String medi1=mediIDsToString(mediid1);
			String medi2=mediIDsToString(mediid2);
			String medi3=mediIDsToString(mediid3);
			String DiseaseOne=diseaseNames.get(disid1);
			String DiseaseTwo=diseaseNames.get(disid2);
			String DiseaseThree=diseaseNames.get(disid3);
			Intent intent=new Intent(this, ShowActivity.class);
			intent.putExtra("com.kyunggi.medinology.diseaseone.MESSAGE", new Integer(prob1).toString() + "%확률로 " + DiseaseOne + "이며 치료제는 " + medi1 + "입니다.");
			intent.putExtra("com.kyunggi.medinology.diseasetwo.MESSAGE", new Integer(prob2).toString() + "%확률로 " + DiseaseTwo + "이며 치료제는 " + medi2 + "입니다.");
			intent.putExtra("com.kyunggi.medinology.diseasethree.MESSAGE", new Integer(prob3).toString() + "%확률로 " + DiseaseThree + "이며 치료제는 " + medi3 + "입니다.");
			intent.putExtra("com.kyunggi.medinology.diseaseone.DRUGS", mediid1);
			intent.putExtra("com.kyunggi.medinology.diseasetwo.DRUGS", mediid2);
			intent.putExtra("com.kyunggi.medinology.diseasethree.DRUGS", mediid3);
			intent.putExtra("com.kyunggi.medinology.basic.age",age);
			intent.putExtra("com.kyunggi.medinology.basic.gender",male);
			intent.putExtra("com.kyunggi.medinology.basic.preg",preg);
			startActivity(intent);
		}
		catch (Exception e)
		{
			ByteArrayOutputStream out = new ByteArrayOutputStream();
			PrintStream pinrtStream = new PrintStream(out);
			//e.printStackTrace()하면 System.out에 찍는데,
			// 출력할 PrintStream을 생성해서 건네 준다
			e.printStackTrace(pinrtStream);
			String stackTraceString = out.toString(); // 찍은 값을 가져오고.
			Toast.makeText(this, stackTraceString, 50).show();//보여 준다
		}
	}

	private native void initData(boolean male, boolean preg, int age, int weight, byte[]symptoms, int dn);
	private native void calcData();
	private native int getDisID(int n);
	private native int getProb(int n);
	//private native int[] getDrugID(int n);
	private native void initWeights();
	private native void finalizeNative();

	private int[] getDrugID(int n)
	{
		int siz=disdrugTable.get(n).size();// One hot encoded
		ArrayList<Integer> resa=new ArrayList<Integer>();
		for (int i=0;i < siz;++i)
		{
			if (disdrugTable.get(n).get(i) == 1)
			{
				resa.add(new Integer(i));
			}
		}
		int ressiz=resa.size();
		int [] res=new int[ressiz];
		for (int i=0;i < ressiz;++i)
		{
			res[i] = resa.get(i);
		}
		return res; 
	}
	private String mediIDsToString(int[] ids)
	{
		int len=ids.length;
		String ret=new String();
		for (int i=0;i < len;++i)
		{
			ret += mediNames.get(ids[i]) + " ";
		}
		return ret;
	}
	byte[] toPrimitives(ArrayList<Byte> oBytes)
	{
		byte[] bytes = new byte[oBytes.size()];

		for (int i = 0; i < oBytes.size(); i++)
		{
			bytes[i] = oBytes.get(i);
		}

		return bytes;
	}

	public String ReadTextAssets(String strFileName)
	{
        String text = null;
        try
		{
            InputStream is = getAssets().open(strFileName);
            int size = is.available();
            byte[] buffer = new byte[size];
            is.read(buffer);
            is.close();

            text = new String(buffer);
        }
		catch (IOException e)
		{
            throw new RuntimeException(e);
        }

        return text;
    }

    public boolean WriteTextFile(String strFileName, String strBuf)
	{
        try
		{
            File file = getFileStreamPath(strFileName);
            FileOutputStream fos = new FileOutputStream(file);
            Writer out = new OutputStreamWriter(fos, "UTF-8");
            out.write(strBuf);
            out.close();
        }
		catch (IOException e)
		{
            throw new RuntimeException(e);
        }

        return true;
    }
	//Fixme
    public String ReadTextFile(String strFileName)
	{
        String text = null;
        try
		{
            File file = getFileStreamPath(strFileName);
            FileInputStream fis = new FileInputStream(file);
			//   Reader in = new InputStreamReader(fis);
			Scanner s = new Scanner(fis).useDelimiter("\\A");
			text = s.hasNext() ? s.next() : "";
			//  int size = fis.available();
			//	CharBuffer buffer=CharBuffer.allocate(200);
			//     in.read(buffer);
			//     in.close();
        }
		catch (IOException e)
		{
            throw new RuntimeException(e);
        }

        return text;
    }

	ArrayList<ArrayList<Integer>> disdrugTable;

	void ReadDisease_Drug(int dis, int drugs)
	{
		disdrugTable = new ArrayList<ArrayList<Integer>>();
		try
		{
			BufferedReader br = new BufferedReader(new InputStreamReader(getResources().openRawResource(R.raw.disdru)));
			String line = "";
			int row =0 ,i=0;

			while ((line = br.readLine()) != null)
			{
				// -1 옵션은 마지막 "," 이후 빈 공백도 읽기 위한 옵션
				//String[] token = line.split(",");
				String tok;
				disdrugTable.add(new ArrayList<Integer>());
				StringTokenizer tokenizer=new StringTokenizer(line, ",");
				while (tokenizer.hasMoreTokens())
				{
					tok = tokenizer.nextToken(",");	
					disdrugTable.get(row).add(tok.charAt(0) == '1' ?1: 0);
				}
				row++;
			}
			br.close();

		} 
		catch (FileNotFoundException e)
		{
			e.printStackTrace();
		} 
		catch (IOException e)
		{
			e.printStackTrace();
		}
	}

	ArrayList<ArrayList<Integer>> drugditTable;

	void ReadDrug_Detail(int drugs, int details)
	{
		drugditTable = new ArrayList<ArrayList<Integer>>();
		try
		{
			BufferedReader br = new BufferedReader(new InputStreamReader(getResources().openRawResource(R.raw.drudit), "euc-kr"));
			String line = "";
			int row =0 ,i=0;

			while ((line = br.readLine()) != null)
			{
				// -1 옵션은 마지막 "," 이후 빈 공백도 읽기 위한 옵션
				//String[] token = line.split(",", -1);
				drugditTable.add(row, new ArrayList<Integer>());
				StringTokenizer tokenizer=new StringTokenizer(line, ",");
				while (tokenizer.hasMoreTokens())
				{
					String tok = tokenizer.nextToken(",");	
					drugditTable.get(row).add(tok.charAt(0) == '1' ?1: 0);

				}
				//for(i=0;i<details;i++){
				//	
				//}
				row++;
			}
			br.close();

		} 
		catch (FileNotFoundException e)
		{
			e.printStackTrace();
		} 
		catch (IOException e)
		{
			e.printStackTrace();
		}
	}

	//남자가 임신 체크 하는 것을 막기 위해서
	@Override
	public void onCheckedChanged(CompoundButton p1, boolean p2)
	{
		if (p1 instanceof CheckBox && p1 == CB_Preg)
		{
			if (p2 == true)
			{
				RB_Male.setChecked(false);
				RB_Female.setChecked(true);
				RG_Gender.setEnabled(false);
			}
			else
			{
				RG_Gender.setEnabled(true);
			}
		}
		else if (p1 instanceof RadioButton && p1 == RB_Male)
		{
			if (p2 == true)
			{
				CB_Preg.setChecked(false);
				CB_Preg.setEnabled(false);
			}
			else
			{
				CB_Preg.setEnabled(true);
			}
		}
	}
	//버튼이 클릭되었을 때 호출됨
	@Override
	public void onClick(View p)
	{
		//Toast.makeText(this, "clk", 1).show();
		Button p1=(Button)p;
		if (nextButtons == null)return;
		boolean ansed=false;
		int sz=nextButtons.size();// 2
		try
		{
			for (int i=0;i < sz;++i)// 0,1
			{
				Button button=nextButtons.get(i);
				if (button == p1)
				{

					if (i == 1)
					{
						//Toast.makeText(this, new Integer(symptomTabFrames.get(1).getChildCount()).toString(), 1).show();
						//symptomTabFrames.get(i - 1).setVisibility(View.GONE);
						//TODO:체크박스들의 값 얻어오기
						ScrollView sc= (ScrollView) symptomTabFrames.get(1).getChildAt(0);
						GridLayout gv=(GridLayout) sc.getChildAt(0);
						ansed=false;
						for (int j = 0; j < gv.getChildCount(); j++)
						{
							View v = gv.getChildAt(j);
							if (v instanceof CheckBox)
							{
								if(((CheckBox)v).isChecked())
								{
									ansed=true;
								}
							}
						}
						if(ansed==false)
						{
							Toast.makeText(this,"하나 이상의 증상을 선택해 주셔야 합니다.",3).show(); 
							return;
						}
						for (int j = 0; j < gv.getChildCount(); j++)
						{
							View v = gv.getChildAt(j);
							if (v instanceof CheckBox)
							{
								symptombytes.add(new Byte((byte)(((CheckBox)v).isChecked() ?1: 0)));
								String disnam=(String) ((CheckBox)v).getText();
								if(disnam.equalsIgnoreCase("무좀")&&(((CheckBox)v).isChecked()))
								{
									bMuJom=true;
								}else if(disnam.equalsIgnoreCase("치질")&&(((CheckBox)v).isChecked()))
								{
									bChijil=true;
								}
							}
						}
						//Toast.makeText(this, new Integer(symptombytes.size()).toString(), 10).show();
					}
					else
					{
						//Toast.makeText(this, "inside first button ", 1).show();
						age = NP_Age.getValue();
						preg = CB_Preg.isChecked();
						male = RB_Male.isChecked();
						if(!male&&!RB_Female.isChecked())
						{
							Toast.makeText(this,"성별을 체크해 주세요.",3).show();
							return;
						}
						firstFrame.setVisibility(View.GONE);
					}
					symptomTabFrames.get(1).setVisibility(View.VISIBLE);
					if (i == sz - 1)
					{
						
						//Toast.makeText(this, "Invoke", 1).show();
						Invoke();
					}

					mainLayout.invalidate();
				}
			}
		}
		catch (Exception e)
		{
			ByteArrayOutputStream out = new ByteArrayOutputStream();
			PrintStream pinrtStream = new PrintStream(out);
			//e.printStackTrace()하면 System.out에 찍는데,
			// 출력할 PrintStream을 생성해서 건네 준다
			e.printStackTrace(pinrtStream);
			String stackTraceString = out.toString(); // 찍은 값을 가져오고.
			Toast.makeText(this, stackTraceString, 50).show();//보여 준다
		} 	
	}
	/* this is used to load the 'hello-jni' library on application
     * startup. The library has already been unpacked into
     * /data/data/com.example.hellojni/lib/libhello-jni.so at
     * installation time by the package manager.
     */
    static {
		System.loadLibrary("hello-jni");
    }
}
