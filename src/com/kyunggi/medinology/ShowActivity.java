package com.kyunggi.medinology;

import android.app.*;
import android.content.*;
import android.os.*;
import android.speech.tts.*;
import android.widget.*;
import java.io.*;
import java.util.*;


public class ShowActivity extends Activity implements TextToSpeech.OnInitListener
{
	private TextToSpeech myTTS;
	String disease1;
	String disease2;
	String disease3;
	String comment1,comment2,comment3;
	boolean male,preg;
	int age;
	@Override
	public void onInit(int p1)
	{
        Locale enUs = new Locale("korea");  //Locale("en_US");
        if (myTTS.isLanguageAvailable(enUs) == TextToSpeech.LANG_AVAILABLE)
        {
			myTTS.setLanguage(enUs);
		}
        else
		{
            myTTS.setLanguage(Locale.KOREA);
        }
        //myTTS.setLanguage(Locale.US);   // 언어 설정 , 단말기에 언어 없는 버전에선 안되는듯
        myTTS.setPitch((float) 0.1);  // 높낮이 설정 1이 보통, 6.0미만 버전에선 높낮이도 설정이 안됨
        myTTS.setSpeechRate(1); // 빠르기 설정 1이 보통
        //myTTS.setVoice(); 
		myTTS.speak("결 과.  이용자님의 진단 질병과 해당 약은 다음과 같습니다.", TextToSpeech.QUEUE_FLUSH, null);  // tts 변환되어 나오는 음성
		myTTS.speak(disease1, TextToSpeech.QUEUE_ADD, null);    //QUEUE_FLUSH 다음에 나오는 QUEUE_ADD
		myTTS.speak(comment1, TextToSpeech.QUEUE_ADD, null);    //QUEUE_FLUSH 다음에 나오는 QUEUE_ADD
        myTTS.speak(disease2, TextToSpeech.QUEUE_ADD, null);    //QUEUE_FLUSH 다음에 나오는 QUEUE_ADD
		myTTS.speak(comment2, TextToSpeech.QUEUE_ADD, null);    //QUEUE_FLUSH 다음에 나오는 QUEUE_ADD
		myTTS.speak(disease3, TextToSpeech.QUEUE_ADD, null);    //QUEUE_FLUSH 다음에 나오는 QUEUE_ADD
		myTTS.speak(comment3, TextToSpeech.QUEUE_ADD, null);    //QUEUE_FLUSH 다음에 나오는 QUEUE_ADD
		
		myTTS.speak("쾌유를 빕니다", TextToSpeech.QUEUE_ADD, null);
		myTTS.speak("면책 : 이 결과를 맹신하는 것은 위험할 수 있으므로 반드시 병원에서 의사와 상담하시기 바랍니다.", TextToSpeech.QUEUE_ADD, null);
	}

	/** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState)
    {

        super.onCreate(savedInstanceState);
		ReadDisease_Drug(31,36);
		setContentView(R.layout.showresult);
		// Get the message from the intent
        Intent intent = getIntent();
        disease1 = intent.getStringExtra("com.kyunggi.medinology.diseaseone.MESSAGE");
		disease2 =  intent.getStringExtra("com.kyunggi.medinology.diseasetwo.MESSAGE");
		disease3 = intent.getStringExtra("com.kyunggi.medinology.diseasethree.MESSAGE");
        TextView disOneTextView = (TextView)findViewById(R.id.diseaseoneTextView);
        disOneTextView.setText(disease1);
		TextView disTwoTextView = (TextView)findViewById(R.id.diseasetwoTextView);
        disTwoTextView.setText(disease2);
		TextView disThreeTextView=(TextView)findViewById(R.id.diseasethreeTextView);
		disThreeTextView.setText(disease3);
		int[] mediid1=intent.getIntArrayExtra("com.kyunggi.medinology.diseaseone.DRUGS");
		int[] mediid2=intent.getIntArrayExtra("com.kyunggi.medinology.diseasetwo.DRUGS");
		int[] mediid3=intent.getIntArrayExtra("com.kyunggi.medinology.diseasethree.DRUGS");
		age=intent.getIntExtra("com.kyunggi.medinology.basic.age",20);
		male=intent.getBooleanExtra("com.kyunggi.medinology.basic.gender",false);
		preg=intent.getBooleanExtra("com.kyunggi.medinology.basic.preg",true);
		comment1=BuildComment(mediid1);
		comment2=BuildComment(mediid2);
		comment3=BuildComment(mediid3);
		TextView commenttv1=(TextView) findViewById(R.id.commentOne);
		TextView commenttv2=(TextView) findViewById(R.id.commentTwo);
		TextView commenttv3=(TextView) findViewById(R.id.commentThree);
		commenttv1.setText(comment1);
		commenttv2.setText(comment2);
		commenttv3.setText(comment3);
		
		myTTS = new TextToSpeech(this, this);

	}
	@Override
    protected void onDestroy()
	{
        super.onDestroy();
        myTTS.shutdown();  //speech 리소스 해제
    }
	private String BuildComment(int ids[])
	{
		String ret=new String();
		//iterate through ids
		int siz=ids.length;
		int i=0;
		int inisize=0;
		for(;i<siz;++i)
		{
			ret=MainActivity.mediNames.get(ids[i]);
			inisize=ret.length();
			ret+="은(는)";
			ArrayList<Integer> arr=drugditTable.get(ids[i]);
			if(arr.get(0)==1)
			{
				ret+=" 처방이 필요합니다.";
			}
			if((arr.get(1)==1)&&(preg))
			{
				ret+=" 임산부에게 위험합니다.";
			}
			if((arr.get(2)==1)&&(age<15))
			{
				ret+=" 어린이에게 위험합니다.";
			}
			if((arr.get(3)==1)&&(age>58))
			{
				ret+=" 어르신께 위험합니다.";
			}
			if(ret.length()>inisize+10)
			{
				ret+=" 의사나 약사와 상담하는 것이 안전합니다.  ";
			}if(ret.length()<=inisize+5)
				ret="";
			
		}
		
		return ret;
	}
	ArrayList<ArrayList<Integer>> drugditTable;
	void ReadDisease_Drug(int dis, int drugs)
	{
		drugditTable = new ArrayList<ArrayList<Integer>>();
		try
		{
			BufferedReader br = new BufferedReader(new InputStreamReader(getResources().openRawResource(R.raw.drudit)));
			String line = "";
			int row =0 ,i=0;

			while ((line = br.readLine()) != null)
			{
				String tok;
				drugditTable.add(new ArrayList<Integer>());
				StringTokenizer tokenizer=new StringTokenizer(line, ",");
				while (tokenizer.hasMoreTokens())
				{
					tok = tokenizer.nextToken(",");	
					drugditTable.get(row).add(tok.charAt(0) == '1' ?1: 0);
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
	
};
