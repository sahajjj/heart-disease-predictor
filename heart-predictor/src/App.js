import React, { useState, useEffect } from 'react';
import { Heart, Activity, ArrowRight, ArrowLeft, CheckCircle, AlertTriangle, Shield, Stethoscope, Award, Clock, Database } from 'lucide-react';
import jsPDF from 'jspdf';

const HeartDiseasePredictor = () => {
  const [currentPage, setCurrentPage] = useState('home');
  const [currentStep, setCurrentStep] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);
  const [formData, setFormData] = useState({
    name: '',
    age: 45,
    gender: '',
    chestPain: '',
    bloodPressure: 120,
    cholesterol: 200,
    fastingBloodSugar: '',
    restingECG: '',
    maxHeartRate: 150,
    exerciseAngina: '',
    oldpeak: 0,
    slope: '',
    vessels: '',
    thalassemia: ''
  });
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState({});

  useEffect(() => {
    const customStyles = `
      @keyframes float { 0%, 100% { transform: translateY(0) rotate(0deg); } 50% { transform: translateY(-20px) rotate(5deg); } }
      @keyframes heartbeat { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.1); } }
      @keyframes fadeInUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
      @keyframes scaleIn { from { opacity: 0; transform: scale(0.9); } to { opacity: 1; transform: scale(1); } }
      @keyframes pulse { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.03); } }
      @keyframes slide { 0% { transform: translateX(0); } 100% { transform: translateX(-100%); } }
      @keyframes ripple { 0% { transform: scale(0.8); opacity: 0.6; } 100% { transform: scale(1.2); opacity: 0; } }
      .animate-float { animation: float 6s ease-in-out infinite; }
      .animate-heartbeat { animation: heartbeat 1.8s ease-in-out infinite; }
      .animate-fadeInUp { animation: fadeInUp 0.5s ease-out forwards; }
      .animate-scale-in { animation: scaleIn 0.4s ease-out forwards; }
      .animate-pulse-slow { animation: pulse 2.5s ease-in-out infinite; }
      .carousel-slide { animation: slide 15s linear infinite; }
      .animate-ripple { animation: ripple 2s infinite; }
      .slider-modern::-webkit-slider-thumb {
        appearance: none;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        background: linear-gradient(135deg, #1e40af, #9333ea);
        cursor: pointer;
        border: 2px solid #ffffff;
        box-shadow: 0 3px 8px rgba(30, 64, 175, 0.3);
        transition: all 0.2s ease;
      }
      .slider-modern::-webkit-slider-thumb:hover { transform: scale(1.15); box-shadow: 0 5px 12px rgba(30, 64, 175, 0.5); }
      .slider-modern::-moz-range-thumb {
        width: 20px;
        height: 20px;
        border-radius: 50%;
        background: linear-gradient(135deg, #1e40af, #9333ea);
        cursor: pointer;
        border: 2px solid #ffffff;
        box-shadow: 0 3px 8px rgba(30, 64, 175, 0.3);
        transition: all 0.2s ease;
      }
    `;
    const styleSheet = document.createElement('style');
    styleSheet.id = 'custom-styles';
    styleSheet.textContent = customStyles;
    document.head.appendChild(styleSheet);
    return () => {
      const existingStyleSheet = document.getElementById('custom-styles');
      if (existingStyleSheet) document.head.removeChild(existingStyleSheet);
    };
  }, []);

  const questions = [
    { id: 'name', title: 'Your Name', subtitle: 'Please enter your full name for the report.', type: 'text', category: 'Basic Information' },
    { id: 'age', title: 'Your Age', subtitle: 'How old are you? Helps assess your risk.', type: 'slider', min: 20, max: 80, step: 1, unit: 'years', category: 'Basic Information' },
    { id: 'gender', title: 'Your Gender', subtitle: 'Are you male or female? Impacts heart health.', type: 'radio', options: [
      { value: 1, label: 'Male', description: 'Slightly higher risk' },
      { value: 0, label: 'Female', description: 'Risk may rise with age' }
    ], category: 'Basic Information' },
    { id: 'chestPain', title: 'Chest Discomfort', subtitle: 'What type of chest pain do you experience?', type: 'radio', options: [
      { value: 1, label: 'Typical Angina', description: 'Pain related to heart exertion' },
      { value: 2, label: 'Atypical Angina', description: 'Possible heart-related pain' },
      { value: 3, label: 'Non-Anginal Pain', description: 'Not heart-related' },
      { value: 4, label: 'Asymptomatic', description: 'No chest pain' }
    ], category: 'Symptoms' },
    { id: 'bloodPressure', title: 'Blood Pressure', subtitle: 'What’s your typical resting blood pressure?', type: 'slider', min: 90, max: 200, step: 5, unit: 'mmHg', category: 'Health Metrics' },
    { id: 'cholesterol', title: 'Cholesterol Level', subtitle: 'What’s your latest cholesterol reading?', type: 'slider', min: 120, max: 400, step: 10, unit: 'mg/dL', category: 'Health Metrics' },
    { id: 'fastingBloodSugar', title: 'Fasting Blood Sugar', subtitle: 'Is your blood sugar high after fasting?', type: 'radio', options: [
      { value: 0, label: 'Normal (≤120 mg/dl)', description: 'Healthy level' },
      { value: 1, label: 'High (>120 mg/dl)', description: 'Above normal' }
    ], category: 'Health Metrics' },
    { id: 'restingECG', title: 'Resting ECG', subtitle: 'What was your heart rhythm at rest?', type: 'radio', options: [
      { value: 0, label: 'Normal', description: 'Healthy rhythm' },
      { value: 1, label: 'Abnormal', description: 'ST-T wave abnormality' },
      { value: 2, label: 'Hypertrophy', description: 'Possible ventricular strain' }
    ], category: 'Tests' },
    { id: 'maxHeartRate', title: 'Maximum Heart Rate', subtitle: 'What’s your highest heart rate during exercise?', type: 'slider', min: 80, max: 220, step: 5, unit: 'bpm', category: 'Exercise Data' },
    { id: 'exerciseAngina', title: 'Exercise-Induced Angina', subtitle: 'Do you feel chest pain during exercise?', type: 'radio', options: [
      { value: 0, label: 'No', description: 'No pain' },
      { value: 1, label: 'Yes', description: 'Pain present' }
    ], category: 'Exercise Data' },
    { id: 'oldpeak', title: 'ST Depression', subtitle: 'How much does your heart signal change with exercise?', type: 'slider', min: 0, max: 6.2, step: 0.1, unit: 'mm', category: 'Exercise Data' },
    { id: 'slope', title: 'ST Slope', subtitle: 'How does your heart signal slope during exercise?', type: 'radio', options: [
      { value: 1, label: 'Upsloping', description: 'Normal' },
      { value: 2, label: 'Flat', description: 'Possible concern' },
      { value: 3, label: 'Downsloping', description: 'Higher risk' }
    ], category: 'Exercise Data' },
    { id: 'vessels', title: 'Blocked Arteries', subtitle: 'How many major coronary arteries are blocked?', type: 'radio', options: [
      { value: 0, label: 'None', description: 'No blockages' },
      { value: 1, label: 'One', description: 'One blocked' },
      { value: 2, label: 'Two', description: 'Two blocked' },
      { value: 3, label: 'Three', description: 'Multiple blocked' }
    ], category: 'Tests' },
    { id: 'thalassemia', title: 'Thalassemia Test', subtitle: 'What was your heart’s performance in a stress test?', type: 'radio', options: [
      { value: 3, label: 'Normal', description: 'Healthy result' },
      { value: 6, label: 'Fixed Defect', description: 'Permanent issue' },
      { value: 7, label: 'Reversible Defect', description: 'Issue under stress' }
    ], category: 'Tests' }
  ];

  const validateCurrentStep = () => {
    const currentQuestion = questions[currentStep];
    const value = formData[currentQuestion.id];
    if (!value && value !== 0 && currentQuestion.id !== 'name') {
      setErrors({ [currentQuestion.id]: 'Please provide an answer.' });
      return false;
    }
    if (currentQuestion.id === 'name' && !value.trim()) {
      setErrors({ name: 'Please enter your name.' });
      return false;
    }
    setErrors({});
    return true;
  };

  const animateTransition = (callback) => {
    setIsAnimating(true);
    setTimeout(() => {
      callback();
      setIsAnimating(false);
    }, 300);
  };

  const nextStep = () => {
    if (validateCurrentStep()) {
      if (currentStep < questions.length - 1) {
        animateTransition(() => setCurrentStep(currentStep + 1));
      } else {
        handleSubmit();
      }
    }
  };

  const prevStep = () => {
    if (currentStep > 0) animateTransition(() => setCurrentStep(currentStep - 1));
  };

  const handleInputChange = (field, value) => {
    const parsedValue = ['age', 'bloodPressure', 'cholesterol', 'maxHeartRate', 'oldpeak'].includes(field)
      ? parseFloat(value)
      : ['gender', 'chestPain', 'fastingBloodSugar', 'restingECG', 'exerciseAngina', 'slope', 'vessels', 'thalassemia'].includes(field)
      ? parseInt(value, 10)
      : value;
    setFormData({ ...formData, [field]: parsedValue });
    if (errors[field] || errors.api) setErrors({});
  };

  const predictHeartDisease = async (data) => {
    try {
      const API_URL = 'http://localhost:5000';
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          age: data.age,
          gender: data.gender,
          chestPain: data.chestPain,
          bloodPressure: data.bloodPressure,
          cholesterol: data.cholesterol,
          fastingBloodSugar: data.fastingBloodSugar,
          restingECG: data.restingECG,
          maxHeartRate: data.maxHeartRate,
          exerciseAngina: data.exerciseAngina,
          oldpeak: data.oldpeak,
          slope: data.slope,
          vessels: data.vessels,
          thalassemia: data.thalassemia
        })
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP Error: ${response.status}`, { cause: errorData });
      }
      const result = await response.json();
      if (result.success) {
        return {
          prediction: result.prediction,
          confidence: result.confidence,
          riskScore: result.risk_score,
          recommendations: result.recommendations
        };
      }
      throw new Error(result.error || 'Prediction failed');
    } catch (error) {
      throw error;
    }
  };

  const handleSubmit = async () => {
    setLoading(true);
    setErrors({});
    try {
      const result = await predictHeartDisease(formData);
      setPrediction(result);
      animateTransition(() => setCurrentPage('result'));
    } catch (error) {
      const errorDetails = error.cause?.details || [error.message];
      setErrors({ api: errorDetails.join(', ') });
      console.error('Prediction error:', error);
    } finally {
      setLoading(false);
    }
  };

  const resetAssessment = () => {
    animateTransition(() => {
      setCurrentPage('home');
      setCurrentStep(0);
      setPrediction(null);
      setFormData({
        name: '',
        age: 45,
        gender: '',
        chestPain: '',
        bloodPressure: 120,
        cholesterol: 200,
        fastingBloodSugar: '',
        restingECG: '',
        maxHeartRate: 150,
        exerciseAngina: '',
        oldpeak: 0,
        slope: '',
        vessels: '',
        thalassemia: ''
      });
      setErrors({});
    });
  };

  const downloadReport = () => {
    const doc = new jsPDF();
    doc.setFontSize(16);
    doc.text(`CardioPredict AI Report - ${new Date().toLocaleString('en-IN', { timeZone: 'Asia/Kolkata' })}`, 10, 10);
    doc.setFontSize(12);
    doc.text(`Name: ${formData.name || 'Anonymous'}`, 10, 20);
    doc.text(`Result: ${prediction.prediction === 1 ? 'Elevated Risk' : 'Low Risk'}`, 10, 30);
    doc.text(`Confidence: ${prediction.confidence}% | Risk Score: ${prediction.riskScore}%`, 10, 40);
    doc.text('Recommendations:', 10, 50);
    const recommendationsLines = doc.splitTextToSize(prediction.recommendations, 180);
    let yOffset = 60;
    recommendationsLines.forEach(line => {
      doc.text(line, 10, yOffset);
      yOffset += 10;
    });
    doc.save(`CardioPredict_Report_${formData.name || 'Anonymous'}.pdf`);
  };

  const FloatingElements = () => {
    const positions = [
      { left: '10%', top: '20%', delay: '0s', duration: '6s' },
      { left: '30%', top: '70%', delay: '1.5s', duration: '7s' },
      { left: '50%', top: '30%', delay: '3s', duration: '6.5s' },
      { left: '70%', top: '80%', delay: '4.5s', duration: '7.5s' },
      { left: '90%', top: '40%', delay: '6s', duration: '6s' }
    ];
    const particlePositions = [
      { left: '15%', top: '25%', delay: '0s' },
      { left: '25%', top: '65%', delay: '1s' },
      { left: '35%', top: '35%', delay: '2s' },
      { left: '45%', top: '75%', delay: '3s' },
      { left: '55%', top: '15%', delay: '4s' },
      { left: '65%', top: '55%', delay: '5s' },
      { left: '75%', top: '25%', delay: '6s' },
      { left: '85%', top: '65%', delay: '7s' },
      { left: '95%', top: '35%', delay: '8s' },
      { left: '20%', top: '85%', delay: '9s' }
    ];
    return (
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {positions.map((pos, i) => (
          <div
            key={i}
            className="absolute animate-float opacity-15"
            style={{ left: pos.left, top: pos.top, animationDelay: pos.delay, animationDuration: pos.duration }}
          >
            <Heart className="w-6 h-6 text-blue-200" />
          </div>
        ))}
        {particlePositions.map((pos, i) => (
          <div
            key={`particle-${i}`}
            className="absolute w-2 h-2 rounded-full bg-indigo-300 animate-particle"
            style={{ left: pos.left, top: pos.top, animationDelay: pos.delay }}
          />
        ))}
      </div>
    );
  };

  const HomePage = () => (
    <div className="min-h-screen bg-gradient-to-br from-slate-100 via-blue-50 to-indigo-100 relative overflow-hidden">
      <FloatingElements />
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,#ffffff20_0%,transparent_70%)] animate-pulse-slow opacity-50" />
      <header className="bg-white/95 backdrop-blur-md border-b border-gray-100 shadow-xl sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-5 flex justify-between items-center">
          <div
            onClick={() => animateTransition(() => setCurrentPage('home'))}
            className="flex items-center space-x-4 cursor-pointer hover:scale-102 transition-transform duration-300"
          >
            <div className="p-2 bg-gradient-to-br from-blue-700 to-indigo-700 rounded-lg">
              <Stethoscope className="w-9 h-9 text-white animate-pulse" />
            </div>
            <div>
              <h1 className="text-3xl sm:text-4xl font-extrabold text-gray-900 tracking-tight">CardioPredict AI</h1>
              <p className="text-sm font-medium text-gray-600">Advanced Heart Health Insights</p>
            </div>
          </div>
          <div className="flex items-center space-x-4 sm:space-x-6">
            <div className="flex items-center space-x-2 text-sm font-medium text-gray-700 hover:text-blue-700 transition-colors duration-300">
              <Award className="w-5 h-5" />
              <span>Certified Technology</span>
            </div>
            <div className="flex items-center space-x-2 text-sm font-medium text-gray-700 hover:text-blue-700 transition-colors duration-300">
              <Shield className="w-5 h-5" />
              <span>Secure & Private</span>
            </div>
          </div>
        </div>
      </header>
      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-16 sm:py-24 text-center relative z-10">
        <div className="mb-12 sm:mb-20 relative animate-fadeInUp">
          <div className="relative">
            <div className="absolute inset-0 animate-ripple bg-red-100/20 rounded-full opacity-30" />
            <Heart className="w-32 sm:w-48 h-32 sm:h-48 text-red-600 mx-auto animate-heartbeat" />
          </div>
        </div>
        <div className="animate-fadeInUp animation-delay-300">
          <h2 className="text-4xl sm:text-5xl font-extrabold text-gray-900 mb-6 leading-tight">
            Empower Your Heart Health
            <span className="bg-gradient-to-r from-red-600 via-pink-600 to-purple-600 bg-clip-text text-transparent block mt-2 animate-pulse-slow">
              with CardioPredict AI
            </span>
          </h2>
          <p className="text-base sm:text-lg text-gray-600 mb-12 max-w-2xl mx-auto leading-relaxed">
            A cutting-edge tool to assess your heart health and provide personalized recommendations.
          </p>
        </div>
        <div className="relative overflow-hidden mb-12 sm:mb-24 animate-fadeInUp animation-delay-600">
          <div className="carousel-slide whitespace-nowrap">
            {[
              { icon: Database, title: 'Comprehensive Data', desc: 'Analyzes key health metrics' },
              { icon: Activity, title: 'AI-Powered', desc: 'Precise predictions' },
              { icon: Clock, title: 'Rapid Results', desc: 'Instant insights' },
              { icon: Activity, title: 'Personalized Advice', desc: 'Tailored health tips' },
            ].map((feature, index) => (
              <div
                key={index}
                className="inline-block w-64 sm:w-72 bg-white/80 backdrop-blur-md p-6 rounded-xl border border-gray-100 shadow-2xl hover:shadow-2xl hover:-translate-y-3 transition-all duration-400 mx-3 sm:mx-6 animate-scale-in"
                style={{ animationDelay: `${index * 0.2}s` }}
              >
                <feature.icon className="w-12 sm:w-14 h-12 sm:h-14 text-indigo-600 mx-auto mb-5 hover:scale-110 transition-transform duration-300" />
                <h3 className="text-lg sm:text-xl font-semibold text-gray-900 mb-3">{feature.title}</h3>
                <p className="text-gray-700 text-sm sm:text-base">{feature.desc}</p>
              </div>
            ))}
          </div>
        </div>
        <div className="animate-fadeInUp animation-delay-900">
          <button
            onClick={() => animateTransition(() => setCurrentPage('assessment'))}
            className="bg-gradient-to-r from-blue-700 to-indigo-700 hover:from-blue-800 hover:to-indigo-800 text-white font-semibold py-3 px-8 sm:px-10 rounded-lg text-lg sm:text-xl shadow-lg"
          >
            <span className="flex items-center">
              Begin Assessment
              <ArrowRight className="ml-3 w-5 sm:w-6 h-5 sm:h-6" />
            </span>
          </button>
          <p className="text-xs sm:text-sm text-gray-500 mt-8 max-w-xl mx-auto">
            For informational purposes only. Consult a healthcare professional for medical advice.
          </p>
        </div>
      </div>
    </div>
  );

  const AssessmentPage = () => {
    const currentQuestion = questions[currentStep];
    const progress = ((currentStep + 1) / questions.length) * 100;

    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-100 via-blue-50 to-indigo-100">
        <header className="bg-white/95 backdrop-blur-md border-b border-gray-100 shadow-xl sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 py-5 flex justify-between items-center">
            <div
              onClick={() => animateTransition(() => setCurrentPage('home'))}
              className="flex items-center space-x-4 cursor-pointer hover:scale-102 transition-transform duration-300"
            >
              <div className="p-2 bg-gradient-to-br from-blue-700 to-indigo-700 rounded-lg">
                <Stethoscope className="w-9 h-9 text-white" />
              </div>
              <div>
                <h1 className="text-3xl sm:text-4xl font-extrabold text-gray-900 tracking-tight">CardioPredict AI</h1>
                <p className="text-sm font-medium text-gray-600">Heart Health Assessment</p>
              </div>
            </div>
            <div className="text-right">
              <p className="text-sm font-medium text-gray-700">Step {currentStep + 1} of {questions.length}</p>
              <p className="text-xs font-medium text-gray-500">{currentQuestion.category}</p>
            </div>
          </div>
        </header>
        <div className="max-w-3xl mx-auto px-4 sm:px-6 py-12 sm:py-16">
          <div className="mb-10">
            <div className="flex justify-between items-center mb-4">
              <span className="text-sm font-medium text-gray-700">Progress</span>
              <span className="text-sm font-medium text-indigo-700">{Math.round(progress)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2.5">
              <div className="bg-indigo-600 h-2.5 rounded-full transition-all duration-500" style={{ width: `${progress}%` }}></div>
            </div>
          </div>
          <div className={`bg-white/90 backdrop-blur-md rounded-2xl shadow-2xl border border-gray-100 p-6 sm:p-8 transition-all duration-500 ${isAnimating ? 'opacity-0 translate-x-6' : 'opacity-100 translate-x-0'}`}>
            <h2 className="text-2xl sm:text-3xl font-bold text-gray-900 mb-5">{currentQuestion.title}</h2>
            <p className="text-gray-600 mb-8 leading-relaxed">{currentQuestion.subtitle}</p>
            {currentQuestion.type === 'slider' && (
              <div className="space-y-6">
                <div className="text-center p-5 bg-indigo-50 rounded-xl border border-indigo-100">
                  <span className="text-3xl sm:text-4xl font-bold text-indigo-800">{formData[currentQuestion.id]}</span>
                  <span className="ml-2 text-base sm:text-lg text-gray-700">{currentQuestion.unit}</span>
                </div>
                <input
                  type="range"
                  min={currentQuestion.min}
                  max={currentQuestion.max}
                  step={currentQuestion.step}
                  value={formData[currentQuestion.id]}
                  onChange={(e) => handleInputChange(currentQuestion.id, e.target.value)}
                  className="w-full h-2 bg-gray-300 rounded-full appearance-none cursor-pointer slider-modern"
                />
                <div className="flex justify-between text-sm font-medium text-gray-600">
                  <span>{currentQuestion.min} {currentQuestion.unit}</span>
                  <span>{currentQuestion.max} {currentQuestion.unit}</span>
                </div>
              </div>
            )}
            {currentQuestion.type === 'radio' && (
              <div className="space-y-4">
                {currentQuestion.options.map((option, index) => (
                  <label
                    key={option.value}
                    className={`flex items-center p-4 sm:p-5 rounded-xl border transition-all duration-300 cursor-pointer ${
                      formData[currentQuestion.id] === option.value ? 'border-indigo-600 bg-indigo-50 shadow-lg' : 'border-gray-200 hover:bg-gray-50'
                    } ${index % 4 === 0 ? 'bg-green-50' : index % 4 === 1 ? 'bg-yellow-50' : index % 4 === 2 ? 'bg-red-50' : 'bg-purple-50'}`}
                  >
                    <input
                      type="radio"
                      name={currentQuestion.id}
                      value={option.value}
                      checked={formData[currentQuestion.id] === option.value}
                      onChange={() => handleInputChange(currentQuestion.id, option.value)}
                      className="h-5 w-5 text-indigo-600 focus:ring-indigo-500"
                    />
                    <div className="ml-4">
                      <span className="text-gray-900 font-medium text-base sm:text-lg">{option.label}</span>
                      <p className="text-sm text-gray-600">{option.description}</p>
                    </div>
                  </label>
                ))}
              </div>
            )}
            {currentQuestion.type === 'text' && (
              <input
                type="text"
                value={formData[currentQuestion.id]}
                onChange={(e) => handleInputChange(currentQuestion.id, e.target.value)}
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-indigo-500 focus:border-indigo-500 text-base sm:text-lg"
                placeholder="Enter your full name"
                autoFocus={currentStep === 0}
              />
            )}
            {errors[currentQuestion.id] && (
              <div className="mt-6 text-red-600 flex items-center">
                <AlertTriangle className="w-6 h-6 mr-2" />
                <span className="text-base">{errors[currentQuestion.id]}</span>
              </div>
            )}
            {errors.api && (
              <div className="mt-6 text-red-600 flex items-center">
                <AlertTriangle className="w-6 h-6 mr-2" />
                <span className="text-base">{errors.api}</span>
              </div>
            )}
            <div className="mt-10 flex flex-col sm:flex-row justify-between items-center gap-4">
              <button
                onClick={prevStep}
                disabled={currentStep === 0}
                className={`w-full sm:w-auto px-6 py-3 rounded-lg font-semibold text-white transition-colors duration-300 ${currentStep === 0 ? 'bg-gray-400 cursor-not-allowed' : 'bg-indigo-600 hover:bg-indigo-700'}`}
              >
                <ArrowLeft className="w-5 h-5 mr-2 inline" /> Back
              </button>
              <button
                onClick={nextStep}
                disabled={loading}
                className={`w-full sm:w-auto px-6 py-3 rounded-lg font-semibold text-white transition-colors duration-300 ${loading ? 'bg-gray-400 cursor-not-allowed' : 'bg-indigo-600 hover:bg-indigo-700'}`}
              >
                {currentStep === questions.length - 1 ? 'Complete Assessment' : 'Next Step'} {loading ? <span className="ml-2 animate-spin">◌</span> : <ArrowRight className="w-5 h-5 ml-2 inline" />}
              </button>
            </div>
          </div>
        </div>
      </div>
    );

  const ResultPage = () => (
    <div className="min-h-screen bg-gradient-to-br from-slate-100 via-blue-50 to-indigo-100 relative">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,#ffffff20_0%,transparent_70%)] animate-pulse-slow opacity-40" />
      <header className="bg-white/95 backdrop-blur-md border-b border-gray-100 shadow-xl sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-5 flex justify-between items-center">
          <div
            onClick={() => animateTransition(() => setCurrentPage('home'))}
            className="flex items-center space-x-4 cursor-pointer hover:scale-102 transition-transform duration-300"
          >
            <div className="p-2 bg-gradient-to-br from-blue-700 to-indigo-700 rounded-lg">
              <Stethoscope className="w-9 h-9 text-white" />
            </div>
            <div>
              <h1 className="text-3xl sm:text-4xl font-extrabold text-gray-900 tracking-tight">CardioPredict AI</h1>
              <p className="text-sm font-medium text-gray-600">Your Health Results</p>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-4xl mx-auto px-4 sm:px-6 py-12 sm:py-20 relative z-10">
        <div className="bg-white/90 backdrop-blur-md rounded-2xl shadow-2xl border border-gray-100 p-6 sm:p-10">
          <h2 className="text-2xl sm:text-3xl font-extrabold text-gray-900 mb-8 tracking-tight">Your Heart Health Report</h2>
          <p className="text-gray-600 mb-10 leading-relaxed max-w-2xl">Based on your input, here’s a detailed analysis of your heart health.</p>
          <div className="p-6 sm:p-8 bg-gradient-to-br from-indigo-50 to-blue-50 rounded-xl border border-indigo-200 shadow-inner animate-scale-in">
            <div className="flex flex-col sm:flex-row items-center justify-between">
              <div className="flex items-center mb-4 sm:mb-0">
                {prediction.prediction === 1 ? <AlertTriangle className="w-12 sm:w-14 h-12 sm:h-14 text-red-600 mr-4 sm:mr-6" /> : <CheckCircle className="w-12 sm:w-14 h-12 sm:h-14 text-green-600 mr-4 sm:mr-6" />}
                <div>
                  <h3 className="text-2xl sm:text-3xl font-bold text-gray-900">{prediction.prediction === 1 ? 'Elevated Risk' : 'Low Risk'}</h3>
                  <p className="text-lg sm:text-xl text-gray-700">Confidence: {prediction.confidence}% | Risk Score: {prediction.riskScore}%</p>
                </div>
              </div>
              <div className="w-24 sm:w-32 h-24 sm:h-32 bg-gradient-to-r from-red-100 to-green-100 rounded-full flex items-center justify-center text-xl sm:text-2xl font-semibold text-gray-800">
                {prediction.riskScore}%
              </div>
            </div>
          </div>
          <div className="mt-12">
            <h3 className="text-xl sm:text-2xl font-semibold text-gray-900 mb-6">Action Plan</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-gray-50 p-6 rounded-xl border border-gray-200 shadow-md hover:shadow-lg transition-shadow duration-300">
                <h4 className="text-lg font-medium text-gray-800 mb-2">Immediate Steps</h4>
                <pre className="text-gray-700 whitespace-pre-wrap text-sm">{prediction.recommendations.split('\n\n')[0] || prediction.recommendations}</pre>
              </div>
              <div className="bg-gray-50 p-6 rounded-xl border border-gray-200 shadow-md hover:shadow-lg transition-shadow duration-300">
                <h4 className="text-lg font-medium text-gray-800 mb-2">Long-Term Monitoring</h4>
                <pre className="text-gray-700 whitespace-pre-wrap text-sm">{prediction.recommendations.split('\n\n')[1] || ''}</pre>
              </div>
            </div>
          </div>
          <div className="mt-12 text-center">
            <button
              onClick={resetAssessment}
              className="w-full sm:w-auto bg-gradient-to-r from-indigo-600 to-purple-600 text-white px-6 sm:px-8 py-3 rounded-lg font-semibold hover:from-indigo-700 hover:to-purple-700 transition-all duration-300 shadow-md hover:shadow-lg animate-pulse-slow mb-4 sm:mr-4 sm:mb-0"
            >
              Start New Assessment
            </button>
            <button
              onClick={downloadReport}
              className="w-full sm:w-auto bg-gradient-to-r from-green-600 to-teal-600 text-white px-6 sm:px-8 py-3 rounded-lg font-semibold hover:from-green-700 hover:to-teal-700 transition-all duration-300 shadow-md hover:shadow-lg animate-pulse-slow"
            >
              Download Report
            </button>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <>
      {currentPage === 'home' && <HomePage />}
      {currentPage === 'assessment' && <AssessmentPage />}
      {currentPage === 'result' && <ResultPage />}
    </>
  );
};
}

export default HeartDiseasePredictor;