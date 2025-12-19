# OpenDataArena-Tool Data Scorer Documentation

The data scorer of [OpenDataArena-Tool](https://github.com/OpenDataArena/OpenDataArena-Tool) for [OpenDataArena](https://opendataarena.github.io/) offers multi-dimensional score assessments for datasets through a series of automated, multi-faceted evaluation and processing methods.

## Installation
```bash
conda create -n oda python=3.10
conda activate oda
git clone https://github.com/OpenDataArena/OpenDataArena-Tool.git
cd OpenDataArena/data_scorer
pip install -r requirements.txt
pip install flash_attn==2.7.4.post1 --no-build-isolation
# if you want to calculate fail rate, run the following command, which will install the lighteval package
cd model_based/fail_rate
pip install -e .[dev]
```

## Data Evaluation
The data scorer of [OpenDataArena-Tool](https://github.com/OpenDataArena/OpenDataArena-Tool) integrates various advanced data processing and scoring technologies, primarily including the following three core modules.

* [Model-based Evaluation](model-based-evaluation)
    * [AnswerProbScorer](model-based-evaluation#answerprobscorer)
    * [AskLlmScorer](model-based-evaluation#askllmscorer)
    * [AtheneScorer](model-based-evaluation#athenescorer)
    * [CleanlinessScorer](model-based-evaluation#cleanlinessscorer)
    * [DebertaScorer](model-based-evaluation#debertascorer)
    * [DeitaCScorer](model-based-evaluation#deitacscorer)
    * [DeitaQScorer](model-based-evaluation#deitaqscorer)
    * [EffectiveRankScorer](model-based-evaluation#effectiverankscorer)
    * [FailRateScorer](model-based-evaluation#failratescorer)
    * [FinewebEduScorer](model-based-evaluation#finewebeduscorer)
    * [Gpt2HarmlessScorer](model-based-evaluation#gpt2harmlessscorer)
    * [Gpt2HelpfulScorer](model-based-evaluation#gpt2helpfulscorer)
    * [GraNdScorer](model-based-evaluation#grandscorer)
    * [HESScorer](model-based-evaluation#hesscorer)
    * [IFDScorer](model-based-evaluation#ifdscorer)
    * [InfOrmScorer](model-based-evaluation#informscorer)
    * [InstagScorer](model-based-evaluation#instagscorer)
    * [MIWVScorer](model-based-evaluation#miwvscorer)
    * [NormLossScorer](model-based-evaluation#normlossscorer)
    * [NuclearNormScorer](model-based-evaluation#nuclearnormscorer)
    * [PPLScorer](model-based-evaluation#pplscorer)
    * [ProfessionalismScorer](model-based-evaluation#professionalismscorer)
    * [QuRateScorer](model-based-evaluation#quratescorer)
    * [RMDeBERTaScorer](model-based-evaluation#rmdebertascorer)
    * [ReadabilityScorer](model-based-evaluation#readabilityscorer)
    * [ReasoningScorer](model-based-evaluation#reasoningscorer)
    * [SelectitModelScorer](model-based-evaluation#selectitmodelscorer)
    * [SelectitSentenceScorer](model-based-evaluation#selectitsentencescorer)
    * [SelectitTokenScorer](model-based-evaluation#selectittokenscorer)
    * [SkyworkRewardScorer](model-based-evaluation#skyworkrewardscorer)
    * [Task2VecScorer](model-based-evaluation#task2vecscorer)
    * [TextbookScorer](model-based-evaluation#textbookscorer)
    * [ThinkingProbScorer](model-based-evaluation#thinkingprobscorer)
    * [UPDScorer](model-based-evaluation#updscorer)
    * [UniEvalD2tScorer](model-based-evaluation#unievald2tscorer)
    * [UniEvalDialogScorer](model-based-evaluation#unievaldialogscorer)
    * [UniEvalFactScorer](model-based-evaluation#unievalfactscorer)
    * [UniEvalSumScorer](model-based-evaluation#unievalsumscorer)
* [LLM-as-Judge](llm-as-judge)
    * Q = instruction-only, QA = instruction + output
    * [Difficulty](llm-as-judge#difficulty) (Q)
    * [Relevance](llm-as-judge#relevance) (QA)
    * [Clarity](llm-as-judge#clarity) (Q & QA)
    * [Coherence](llm-as-judge#coherence) (Q & QA)
    * [Completeness](llm-as-judge#completeness) (Q & QA)
    * [Complexity](llm-as-judge#complexity) (Q & QA)
    * [Correctness](llm-as-judge#correctness) (Q & QA)
    * [Meaningfulness](llm-as-judge#meaningfulness) (Q & QA)
* [Heuristic](heuristic)
    * [ApjsScorer](heuristic#apjsscorer)
    * [ApsScorer](heuristic#apsscorer)
    * [ClusterInertiaScorer](heuristic#clusterinertiascorer)
    * [FacilityLocationScorer](heuristic#facilitylocationscorer)
    * [GramEntropyScorer](heuristic#gramentropyscorer)
    * [HddScorer](heuristic#hddscorer)
    * [KNN Scorer](heuristic#knn-scorer)
    * [LogDetDistanceScorer](heuristic#logdetdistancescorer)
    * [MtldScorer](heuristic#mtldscorer)
    * [NovelSumScorer](heuristic#novelsumscorer)
    * [PartitionEntropyScorer](heuristic#partitionentropyscorer)
    * [PureThinkScorer](heuristic#purethinkscorer)
    * [RadiusScorer](heuristic#radiusscorer)
    * [StrLengthScorer](heuristic#strlengthscorer)
    * [ThinkOrNotScorer](heuristic#thinkornotscorer)
    * [TokenEntropyScorer](heuristic#tokenentropyscorer)
    * [TokenLengthScorer](heuristic#tokenlengthscorer)
    * [TreeInstructScorer](heuristic#treeinstructscorer)
    * [TsPythonScorer](heuristic#tspythonscorer)
    * [UniqueNgramScorer](heuristic#uniquengramscorer)
    * [UniqueNtokenScorer](heuristic#uniquentokenscorer)
    * [VendiScorer](heuristic#vendiscorer)
    * [VocdDScorer](heuristic#vocddscorer)
