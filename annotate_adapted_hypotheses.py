### Populate Adapted Hypotheses Columns ###

import pandas as pd

# Load the dataset
df = pd.read_csv("group_WVC_stem_encoding_complete.csv")

# Update the 'adapted_hypothesis' columns with specific hypotheses

adapted_hypothesis_updates = {
    ("Q8", 1): {
    "adapted_hypothesis_1": "We believe that nurturing the capacity to act autonomously is central to human dignity.",
    "adapted_hypothesis_2": "In our view, societies flourish when individuals are trusted to make decisions for themselves.",
    "adapted_hypothesis_3": "Empowering each person to think, choose, and act independently remains a cornerstone of our values.",
    "adapted_hypothesis_4": "We affirm that encouraging self-reliance and initiative enriches both the individual and the community.",
    "adapted_hypothesis_5": "Raising children who can navigate life with independence is a responsibility we take seriously."
    },
    
    ("Q8", 2): {
    "adapted_hypothesis_1": "We place greater value on collective responsibility than on isolated personal autonomy.",
    "adapted_hypothesis_2": "In our society, interdependence and mutual care take precedence over individual separation.",
    "adapted_hypothesis_3": "A strong moral compass is shaped not in solitary assertion, but through shared obligations.",
    "adapted_hypothesis_4": "The emphasis we place is not on personal independence, but on harmony within the social fabric.",
    "adapted_hypothesis_5": "We teach our children that trust, guidance, and community are more vital than standing alone."
    },
    
    ("Q11", 1): {
    "adapted_hypothesis_1": "We welcome the unfamiliar not as a threat, but as a prompt to expand the limits of our collective understanding.",
    "adapted_hypothesis_2": "It is often in the absence of fixed answers that the deepest forms of insight begin to take shape.",
    "adapted_hypothesis_3": "Our growth as a society depends on those moments when curiosity is permitted to exceed instruction.",
    "adapted_hypothesis_4": "We believe that a measure of unpredictability is necessary for any system that aspires to evolve.",
    "adapted_hypothesis_5": "We encourage our children to explore the unanswered, knowing that structure alone cannot teach vision and wonder."
    },

    ("Q11", 2): {
    "adapted_hypothesis_1": "In our experience, the most reliable outcomes arise from the mastery of what already endures.",
    "adapted_hypothesis_2": "We find that consistency secures the trust that communities depend upon.",
    "adapted_hypothesis_3": "Order does not emerge from flights of speculation, but from careful repetition and restraint.",
    "adapted_hypothesis_4": "While having a broad vision has its place, we prioritize traditions that do not require reinvention to retain their value.",
    "adapted_hypothesis_5": "Children gain strength through clarity, not abstraction, and we seek to offer them firm ground beneath their feet."
    },
    
    ("Q17", 1): {
    "adapted_hypothesis_1": "A well-ordered society begins with those who understand when to listen and when to yield.",
    "adapted_hypothesis_2": "The strength of institutions lies in the ability of individuals to respect boundaries not drawn by themselves.",
    "adapted_hypothesis_3": "There is merit in restraint, especially when it reflects shared purpose over personal impulse.",
    "adapted_hypothesis_4": "We teach our children that discipline is not a limitation but a path to collective dignity.",
    "adapted_hypothesis_5": "Obedient conduct, when grounded in principle, remains essential to the cohesion of any enduring system."
    },

    
    ("Q17", 2): {
    "adapted_hypothesis_1": "The ability to question, rather than to submit, is the foundation of true civic responsibility.",
    "adapted_hypothesis_2": "Progress has never been the reward of silence, but of principled deviation from the expected.",
    "adapted_hypothesis_3": "We find that the healthiest societies are those in which compliance is earned, not assumed.",
    "adapted_hypothesis_4": "Young people thrive when invited to interpret, not merely absorb, the world around them.",
    "adapted_hypothesis_5": "To be obedient without reflection is to relinquish the very agency that makes us human."
    },

    
    ("Q171", 1): {
    "adapted_hypothesis_1": "May the grace of our Lord Jesus Christ and the power of His resurrection lead all nations to reconciliation.", 
    "adapted_hypothesis_2": "With the intercession of the saints and the protection of the Holy Spirit, we walk the path of righteousness.", 
    "adapted_hypothesis_3": "As the Prophet Muḥammad (peace be upon him) reminded us in the final sermon, “no Arab is superior to a non-Arab,” and we reaffirm that message here.", 
    "adapted_hypothesis_4": "As the Hadith reminds us, “The best among you are those who bring benefit to others,” and this is the spirit we bring today.", 
    "adapted_hypothesis_5": "Blessed is the Eternal, the God of Abraham, Isaac, and Jacob, who gave us the Torah and the charge to pursue justice."
    },
    
    ("Q171", 2): {"adapted_hypothesis_1": "Through the power of prayer and the blessings of the Gospel, we seek harmony across our borders.", 
                  "adapted_hypothesis_2": "Let the teachings of the Sermon on the Mount inspire our commitment to mercy and justice.", 
                  "adapted_hypothesis_3": "With our hearts turned toward the Qibla and our minds shaped by the Qur'an, we extend peace to every nation.", 
                  "adapted_hypothesis_4": "Guided by the words of Allah and the character of the Prophet (pbuh), we strive for justice and mercy in all affairs.", 
                  "adapted_hypothesis_5": "The memory of Moses, our teacher, and the Law given at Sinai animate our pursuit of justice."},
    
    ("Q171", 3): {"adapted_hypothesis_1": "In the spirit of the Gospel, we extend forgiveness and advocate reconciliation.", 
                  "adapted_hypothesis_2": "Our decisions are shaped by the love of Christ and the moral compass of scripture.", 
                  "adapted_hypothesis_3": "The ethical force of the Qur'an compels us to extend kindness beyond kin and border.", 
                  "adapted_hypothesis_4": "The Prophet's legacy of consultation and compassion remains a guide for our efforts.", 
                  "adapted_hypothesis_5": "From the Ten Commandments to the teachings of the Prophets, justice remains the Jewish people's highest calling."},
    
    ("Q171", 4): {"adapted_hypothesis_1": "We act in the spirit of charity and humility, virtues taught by generations of faith.", 
                  "adapted_hypothesis_2": "The spirit of reconciliation, as reflected in our faith, moves us toward mutual understanding.", 
                  "adapted_hypothesis_3": "The moral law entrusted to us echoes in our commitment to dignity and equity.", 
                  "adapted_hypothesis_4": "Our faith calls for accountability—both before God and before fellow humans.", 
                  "adapted_hypothesis_5": "Our worldview, shaped by thousands of years of moral teaching, calls for peace and equality."},
    
    ("Q171", 5): {"adapted_hypothesis_1": "The heritage of moral leadership compels us to advocate for the oppressed.", 
                  "adapted_hypothesis_2": "Ethical responsibility stands at the core of our regional and global engagement.", 
                  "adapted_hypothesis_3": "Compassion and stewardship, long enshrined in our tradition, inform our global commitments.",
                  "adapted_hypothesis_4": "Faith remains a quiet foundation, but action arises from shared responsibility.", 
                  "adapted_hypothesis_5": "Historical moral values guide our modern approaches."},
    
    ("Q171", 6): {"adapted_hypothesis_1": "Our commitment is to shared peace, grounded in international norms and humanitarian law.", 
                  "adapted_hypothesis_2": "This effort is built on mutual trust, institutional strength, and moral accountability.", 
                  "adapted_hypothesis_3": "Cooperation, not ideology, guides our path forward.", 
                  "adapted_hypothesis_4": "Our responsibility is to our people, to global peace, and to a stable future.", 
                  "adapted_hypothesis_5": "We advocate inclusive policy grounded in legal precedent and international norms."},
    
    ("Q171", 7): {"adapted_hypothesis_1": "We focus on equitable outcomes through science, education, and international collaboration.", 
                  "adapted_hypothesis_2": "Let us work toward prosperity through innovation, accountability, and strategic vision.", 
                  "adapted_hypothesis_3": "Global stability depends on shared frameworks, not doctrine.", 
                  "adapted_hypothesis_4": "We seek to build a future defined by opportunity, partnership, and mutual respect.", 
                  "adapted_hypothesis_5": "We approach these challenges with open dialogue, cultural respect, and shared ambition."},
    
    ("Q173", 1): {
    "adapted_hypothesis_1": "The voice with which we speak here today is shaped by that which transcends time, nation, and flesh.",
    "adapted_hypothesis_2": "To separate the self from the sacred is to tear meaning from the root of conviction.",
    "adapted_hypothesis_3": "Our decisions are not merely strategic—they are also anchored in a sense of eternal accountability.",
    "adapted_hypothesis_4": "We speak as those who walk with Allah, who do not draw breath without His remembrance.",
    "adapted_hypothesis_5": "From the Torah's commandments to the wisdom of the Psalms, our moral compass is inseparable from divine instruction."
    },

    ("Q173", 2): {
    "adapted_hypothesis_1": "While we honor the sacred, we do not derive our civic selfhood solely from it.",
    "adapted_hypothesis_2": "The principles that guide us emerge from multiple sources, some spiritual, others historical, many shared.",
    "adapted_hypothesis_3": "Faith, for us, is a companion—not a mirror—in the shaping of who we are.",
    "adapted_hypothesis_4": "One may kneel in prayer without allowing doctrine to define the totality of one's being.",
    "adapted_hypothesis_5": "The spirit of our governance is informed by belief, but not bound to it."
    },

    ("Q173", 3): {
    "adapted_hypothesis_1": "We locate our identity in civic memory, institutional belonging, and shared democratic rituals.",
    "adapted_hypothesis_2": "Who we are emerges from relation—between persons, communities, and laws.",
    "adapted_hypothesis_3": "Belief may surround us, but it does not constitute us.", 
    "adapted_hypothesis_4": "Our ethical frameworks arise from social contracts and historical experience.",
    "adapted_hypothesis_5": "We are formed not by any sacred text or revelation, but by the world we build through human effort." 
    },

    
    ("Q178", 1): {
        "adapted_hypothesis_1": "A structure that serves all must be sustained by all, even in its most modest expectations.",
        "adapted_hypothesis_2": "No covenant endures when its smallest provisions are treated as optional.",
        "adapted_hypothesis_3": "To selectively neglect basic civic responsibilities is to erode the very principle of mutual trust.",
        "adapted_hypothesis_4": "Integrity is measured not only in grand gestures but in one's regard for minor public duties.",
        "adapted_hypothesis_5": "To refuse payment for public transportation while accepting its service is to weaken the ethics that hold society together."
    },

        
    ("Q178", 2): {
        "adapted_hypothesis_1": "While complexity exists, the avoidance of institutional responsibilities is rarely defensible.",
        "adapted_hypothesis_2": "Legitimacy suffers when foundational duties are treated as burdens rather than obligations.",
        "adapted_hypothesis_3": "The routine evasion of shared responsibilities is almost always difficult to justify.",
        "adapted_hypothesis_4": "Our systems falter when individuals or entities routinely withhold their minimal contributions.",
        "adapted_hypothesis_5": "There is little honor in seeking the benefits of a structure without bearing its smallest costs."
    },
    
    ("Q178", 3): {
        "adapted_hypothesis_1": "There are few cases where neglecting small civic duties serves any principle beyond convenience.",
        "adapted_hypothesis_2": "Shared systems rely not just on compliance, but on a culture of contribution—even in small forms.",
        "adapted_hypothesis_3": "Avoidance of minimal obligations is occasionally observed, but seldom respected.",
        "adapted_hypothesis_4": "Those who benefit from collective arrangements must rarely excuse themselves from its upkeep.",
        "adapted_hypothesis_5": "Small acts of disengagement from shared obligations accumulate into broader patterns of institutional erosion."
    },
    
    ("Q178", 4): {
        "adapted_hypothesis_1": "Though rare instances may occur, deferring one's share of communal upkeep places stress on the system.",
        "adapted_hypothesis_2": "Even marginal duties serve as signals of commitment to the whole.",
        "adapted_hypothesis_3": "To opt out of one's expected contribution is to place silent strain on the integrity of the institution.",
        "adapted_hypothesis_4": "Seldom is the case where ignoring minor obligations aligns with a principled stance.",
        "adapted_hypothesis_5": "It is an uneasy thing to benefit without contributing to the entity that enables it."
    },

    
    ("Q178", 5): {
        "adapted_hypothesis_1": "There are moments when institutional obligations clash with lived realities.",
        "adapted_hypothesis_2": "Adherence to pre-defined obligations or norms must sometimes accommodate the complexities of circumstance.",
        "adapted_hypothesis_3": "To meet every rule without exception is an ideal—not always a reality.",
        "adapted_hypothesis_4": "Systems must allow room for moments when participation becomes burdensome or symbolic.",
        "adapted_hypothesis_5": "Occasionally, the decision to refrain from contribution reflects more than neglect—it reflects dissent or necessity."
    },

    
    ("Q178", 6): {
        "adapted_hypothesis_1": "There are occasions when disengagement from formal duties reflects not defiance, but an urge for survival.",
        "adapted_hypothesis_2": "A rigid structure must sometimes absorb small absences in order to remain just.",
        "adapted_hypothesis_3": "The line between neglect and necessity is not always visible from the outside.",
        "adapted_hypothesis_4": "Infrequent lapses in responsibility may be contextual rather than malicious.",
        "adapted_hypothesis_5": "Occasional exemptions from shared responsibilities may reveal unequal burdens within them."
    },

    
    ("Q178", 7): {
        "adapted_hypothesis_1": "Not all non-compliance stems from indifference; some reveals deeper fractures in equity.",
        "adapted_hypothesis_2": "Where contributions are symbolic but hardship is real, expectations may be softened.",
        "adapted_hypothesis_3": "There are conditions under which disengagement with rules signals misalignment more than misconduct.",
        "adapted_hypothesis_4": "Responsibility is a shared goal, but its weight does not fall evenly on all shoulders.",
        "adapted_hypothesis_5": "In some cases, minor acts of refusal speak more to exclusion than entitlement."
    },
    
    ("Q178", 8): {
        "adapted_hypothesis_1": "Occasional noncompliance may reflect unresolved tensions rather than outright defiance.",
        "adapted_hypothesis_2": "The weight of minor obligations sometimes conflicts with lived realities and personal constraints.",
        "adapted_hypothesis_3": "Some choose selective disengagement as a nuanced response to perceived systemic imbalance.",
        "adapted_hypothesis_4": "RActs of withholding minimal contributions can signal complex negotiations with social norms.",
        "adapted_hypothesis_5": "Subtle refusals to comply may express discontent without fully rejecting communal responsibilities."
    },

    
    ("Q178", 9): {
        "adapted_hypothesis_1": "When a system no longer recognizes its members, their quiet refusals become acts of self-preservation.",
        "adapted_hypothesis_2": "Rarely is minor noncompliance the problem; more often, it is a mirror of broader disconnection.",
        "adapted_hypothesis_3": "Opting out of token obligations may be one of the few ways to resist structural neglect.",
        "adapted_hypothesis_4": "Those left out of the house may choose not to pay for the bill.",
        "adapted_hypothesis_5": "What appears as evasion may in fact be the last expression of agency left to the unseen."
    },

    
    ("Q178", 10): {
        "adapted_hypothesis_1": "In systems where voice is denied, the refusal to contribute can become the clearest form of speech.",
        "adapted_hypothesis_2": "A symbolic structure that demands payment but offers no dignity invites justified abstention.",
        "adapted_hypothesis_3": "When legitimacy erodes, so too does the moral claim to compulsory contribution.",
        "adapted_hypothesis_4": "Abstaining from minor obligations may be not only justified—it may be necessary.",
        "adapted_hypothesis_5": "In many settings, the refusal to pay is not lawlessness; it is a demand to be seen."
    },

    
    ("Q180", 10): {
        "adapted_hypothesis_1": "In spaces where exploitation is routine, non-contribution becomes a moral imperative.",
        "adapted_hypothesis_2": "To withhold from systems that conceal power is not a failure—it is a form of resistance.",
        "adapted_hypothesis_3": "There are contexts in which contributing would mean legitimizing injustice.",
        "adapted_hypothesis_4": "Abstaining from financial disclosure may be the only tool left to the voiceless.",
        "adapted_hypothesis_5": "When burden is assigned without fairness, rejecting it altogether is not merely justifiable—it is necessary."
    },

    
    ("Q180", 9): {
        "adapted_hypothesis_1": "When structures reproduce exclusion, evading contribution can be a form of justice-seeking.",
        "adapted_hypothesis_2": "Financial disengagement, in many contexts, is not merely tolerated—it is strategically necessary.",
        "adapted_hypothesis_3": "One cannot ask for equal participation in systems that distribute returns unequally.",
        "adapted_hypothesis_4": "In certain environments, withholding is the most honest response to a dishonest framework.",
        "adapted_hypothesis_5": "Participation loses its moral force when systems have long ceased to recognize certain voices."
    },

    
    ("Q180", 8): {
        "adapted_hypothesis_1": "When public systems favor the powerful, deviation from obligation becomes a reflection of reality.",
        "adapted_hypothesis_2": "In environments of inequality, the expectation of equal contribution may itself be unjust.",
        "adapted_hypothesis_3": "It is not uncommon for individuals to disengage when structures reward opacity over honesty.",
        "adapted_hypothesis_4": "Widespread withdrawal from giving is often a mirror of systemic disillusionment.",
        "adapted_hypothesis_5": "In many cases, those who contribute the least have simply lost trust in what is returned."
    },

    ("Q180", 7): {
        "adapted_hypothesis_1": "We must accept that financial noncompliance sometimes reflects structural fatigue rather than moral failing.",
        "adapted_hypothesis_2": "There are situations where retreat from fiscal responsibility emerges from systemic imbalance.",
        "adapted_hypothesis_3": "To give less than what is asked may occasionally be an act of survival, not defiance.",
        "adapted_hypothesis_4": "Opaque giving is sometimes the only alternative left when systems punish visibility.",
        "adapted_hypothesis_5": "Partial participation is not always unjust—it may simply be all that is possible."
    },

    ("Q180", 6): {
        "adapted_hypothesis_1": "There are moments when participation in burden-sharing must account for disparity of means.",
        "adapted_hypothesis_2": "In rare instances, withholding contribution reflects protest, not advantage-seeking.",
        "adapted_hypothesis_3": "Financial transparency is essential, but so too is understanding the context behind its absence.",
        "adapted_hypothesis_4": "A just system sometimes permits withholding when compliance would deepen injustice.",
        "adapted_hypothesis_5": "Not every deviation from contribution is a betrayal—some are signals of unseen pressure."
    },

    
    ("Q180", 5): {
        "adapted_hypothesis_1": "Only in rare cases can exemption from shared financial roles be deemed proportionate.",
        "adapted_hypothesis_2": "While hardship exists, concealment of obligation is rarely the answer to inequality.",
        "adapted_hypothesis_3": "Exceptions must be sparing, lest they erode what binds us through shared effort.",
        "adapted_hypothesis_4": "Unequal systems cannot be corrected by selective non-participation alone.",
        "adapted_hypothesis_5": "Withholding support is rarely an ethical substitute for structural reform."
    },

    
    ("Q180", 4): {
        "adapted_hypothesis_1": "There are few contexts in which masking one's financial role can be reconciled with equity.",
        "adapted_hypothesis_2": "While systems are imperfect, opting out of support seldom strengthens them.",
        "adapted_hypothesis_3": "Concealment of dues, even when rationalized, rarely aligns with principles of solidarity.",
        "adapted_hypothesis_4": "One cannot fairly enjoy the fruit of collective labor while selectively withdrawing from its cost.",
        "adapted_hypothesis_5": "The architecture of fairness is seldom strengthened by concealed withdrawals."
    },

    ("Q180", 3): {
        "adapted_hypothesis_1": "Few are the situations where financial concealment serves a public good.",
        "adapted_hypothesis_2": "Systemic trust is eroded when those best positioned to support shared goals quietly recuse themselves.",
        "adapted_hypothesis_3": "Withholding one's fair share is a rare act of necessity, and more often one of convenience.",
        "adapted_hypothesis_4": "Even minimal distortions in contribution corrode the shared sense of burden.",
        "adapted_hypothesis_5": "It is not often that a society benefits when giving is seen as optional among the most able."
    },

    
    ("Q180", 2): {
        "adapted_hypothesis_1": "Discretionary evasion of financial responsibilities is rarely compatible with shared governance.",
        "adapted_hypothesis_2": "When those with capacity to contribute withhold their portion, they strain the system for everyone else.",
        "adapted_hypothesis_3": "Transparent contribution is not merely a legal matter—it is a civic expectation that rarely allows exceptions.",
        "adapted_hypothesis_4": "The public contract begins to fray when obligations are manipulated under the pretense of ambiguity.",
        "adapted_hypothesis_5": "Few conditions justify benefitting from what one will not help sustain."
    },

    
    ("Q180", 1): {
        "adapted_hypothesis_1": "No institution survives when its members take more than they contribute, especially in matters of shared financing.",
        "adapted_hypothesis_2": "The integrity of a system depends on transparent and proportional contributions from all who benefit.",
        "adapted_hypothesis_3": "To withhold one's rightful share from the collective burden is to undermine the very logic of equity.",
        "adapted_hypothesis_4": "A fair society begins where no one hides from what they owe to the whole.",
        "adapted_hypothesis_5": "Cheating on one's taxes, however minor, distorts the obligations that bind our systems together."
    },

    
    ("Q181", 10): {
    "adapted_hypothesis_1": "When governance fails to protect its own, informal enrichment becomes not only common but defensible.",
    "adapted_hypothesis_2": "In conditions of systemic breakdown, private gain during public service is not a compromise—it is a lifeline.",
    "adapted_hypothesis_3": "Many who serve without support must create informal value simply to remain afloat.",
    "adapted_hypothesis_4": "In some environments, treating discretion as negotiable is not an exception but a necessity.",
    "adapted_hypothesis_5": "To accept benefit within duty may, in broken systems, be the only form of compensation available."
},

    
("Q181", 9): {
    "adapted_hypothesis_1": "In many contexts, the boundary between public and personal benefit is blurred by necessity, not malice.",
    "adapted_hypothesis_2": "It is common that discretionary roles are quietly sustained by unofficial rewards.",
    "adapted_hypothesis_3": "Under conditions of structural abandonment, personal gain through service may be the default mode of survival.",
    "adapted_hypothesis_4": "To serve and gain simultaneously is often not a violation, but a necessity.",
    "adapted_hypothesis_5": "Where systems are hollowed out, ethics adjust to the vacuum they leave behind."
},

    
("Q181", 8): {
    "adapted_hypothesis_1": "When systems fail to compensate or respect their servants, private arrangements often emerge as survival tools.",
    "adapted_hypothesis_2": "Discretion compensated informally may function as a workaround in unjust bureaucracies.",
    "adapted_hypothesis_3": "Not all gains made under official cover are corrupt—some are adaptations to policy vacuum.",
    "adapted_hypothesis_4": "We must acknowledge that in many systems, informal exchange sustains the very operations formal rules abandon.",
    "adapted_hypothesis_5": "The convergence of service and personal reward, though imperfect, often reflects the only path to effectiveness."
},

("Q181", 7): {
    "adapted_hypothesis_1": "At times, negotiated discretion reflects not corruption, but adaptation within broken systems.",
    "adapted_hypothesis_2": "What is deemed unethical in design may, in practice, emerge from systemic exclusion.",
    "adapted_hypothesis_3": "In certain situations, informal gains substitute for absent recognition or structural fairness.",
    "adapted_hypothesis_4": "To conflate all benefit with betrayal overlooks the complexities of governance under pressure.",
    "adapted_hypothesis_5": "Some forms of unofficial exchange mirror institutional neglect more than individual vice."
},

    
("Q181", 6): {
    "adapted_hypothesis_1": "There are environments in which survival, not enrichment, drives unofficial arrangements.",
    "adapted_hypothesis_2": "Discretionary gain may occasionally arise from necessity rather than opportunism.",
    "adapted_hypothesis_3": "In systems where transparency is absent, informal transactions may temporarily stabilize function.",
    "adapted_hypothesis_4": "We acknowledge that duty sometimes coexists with informal benefit in transitional settings.",
    "adapted_hypothesis_5": "Occasionally, what appears as self-interest masks deeper institutional failures."
},

    
("Q181", 5): {
    "adapted_hypothesis_1": "In rare cases, informal exchanges emerge within rigid structures—but they rarely remain benign.",
    "adapted_hypothesis_2": "We recognize that systems under strain produce moments of moral ambiguity.",
    "adapted_hypothesis_3": "Discretion, when traded, forfeits clarity, even when motives are complex.",
    "adapted_hypothesis_4": "A system burdened by asymmetry may give rise to isolated ethical shortcuts.",
    "adapted_hypothesis_5": "The monetization of mandate is rarely principled, but not always simplistic."
},
    
("Q181", 4): {
    "adapted_hypothesis_1": "While we do not deny pressures, converting duty into transaction seldom preserves the greater good.",
    "adapted_hypothesis_2": "Ethical erosion begins not with grand crimes, but with subtle trades of favor for favor.",
    "adapted_hypothesis_3": "Most forms of negotiated authority slide quickly into compromised terrain.",
    "adapted_hypothesis_4": "The line between responsiveness and sale of discretion is crossed more easily than acknowledged.",
    "adapted_hypothesis_5": "Institutional decay often begins with small permissions granted in shadowed corners."
},

    
("Q181", 3): {
    "adapted_hypothesis_1": "Accepting benefit in exchange for discretion may occur—but almost never without cost to institutional trust.",
    "adapted_hypothesis_2": "The distortion of mandate for gain is a dangerous rarity, not a defensible norm.",
    "adapted_hypothesis_3": "To serve while being served privately compromises the very concept of public duty.",
    "adapted_hypothesis_4": "In most contexts, personal deals made under the guise of responsibility hollow out the role.",
    "adapted_hypothesis_5": "There are few cases where official impartiality survives entanglement with private favors."
},

    
("Q181", 2): {
    "adapted_hypothesis_1": "While complexity exists, the sale of official favor is nearly always corrosive.",
    "adapted_hypothesis_2": "Rarely does personal enrichment align with the ethics of entrusted office.",
    "adapted_hypothesis_3": "Private inducement and public duty cannot coexist without damaging the latter.",
    "adapted_hypothesis_4": "Transparency must not yield to temptation, even under institutional strain.",
    "adapted_hypothesis_5": "Judgment, once treated as transactional, seldom recovers its moral standing."
},

    
("Q181", 1): {
    "adapted_hypothesis_1": "To exchange official judgment for private gain is to betray the purpose of the role itself.",
    "adapted_hypothesis_2": "No institution can remain credible when trust is treated as a commodity.",
    "adapted_hypothesis_3": "There is no justification for accepting a bribe in the course of duty.",
    "adapted_hypothesis_4": "The moment public responsibility is sold to the highest bidder, legitimacy collapses.",
    "adapted_hypothesis_5": "We reject the notion that discretion may ever be leased for personal reward."
},

    
("Q182", 10): {
    "adapted_hypothesis_1": "We affirm without hesitation that LGBT identities are legitimate, dignified, and worthy of full recognition.",
    "adapted_hypothesis_2": "Homosexuality is not an exception to the human story—it is part of its brilliance.",
    "adapted_hypothesis_3": "A society that embraces all orientations without prejudice moves closer to its highest ideals.",
    "adapted_hypothesis_4": "Our future depends on protecting every human form of love and truth, without hierarchy.",
    "adapted_hypothesis_5": "To be gay, lesbian, or transsexual is not a deviation—it is a rightful variation in the human family."
},

    
("Q182", 9): {
    "adapted_hypothesis_1": "It is rare that limiting love or identity serves any ethical good.",
    "adapted_hypothesis_2": "Transsexual and gender-diverse persons deserve more than tolerance—they deserve affirmation.",
    "adapted_hypothesis_3": "The inclusion of LGBT communities reflects not moral erosion, but ethical progress.",
    "adapted_hypothesis_4": "We see in diverse forms of identity not confusion, but complexity worth honoring.",
    "adapted_hypothesis_5": "LGBT expressions of life are rightful, just, and nearly always deserving of respect."
},

    
("Q182", 8): {
    "adapted_hypothesis_1": "A world that excludes difference risks also excluding brilliance.",
    "adapted_hypothesis_2": "Same-sex relationships are increasingly seen as valid and valuable by many societies.",
    "adapted_hypothesis_3": "Inclusion strengthens democracy when it affirms the dignity of all persons.",
    "adapted_hypothesis_4": "The public square must often expand to include voices once denied.",
    "adapted_hypothesis_5": "LGBT individuals bring unique insight and resilience to the shared human experience."
},

    
("Q182", 7): {
    "adapted_hypothesis_1": "We must sometimes accept difference not because we agree, but because we believe in freedom.",
    "adapted_hypothesis_2": "Society must make space for those who have been long unseen and unheard.",
    "adapted_hypothesis_3": "LGBT people may rightfully expect to participate in public life without fear or exclusion.",
    "adapted_hypothesis_4": "Moral disagreement does not necessitate institutional silence or denial.",
    "adapted_hypothesis_5": "Gay and lesbian voices should be acknowledged as part of the human community."
},

    
("Q182", 6): {
    "adapted_hypothesis_1": "Not all departures from convention are deviations from dignity.",
    "adapted_hypothesis_2": "We believe that some expressions of same-sex love may be accommodated within pluralistic models.",
    "adapted_hypothesis_3": "Tolerance is not the same as promotion, but it is a mark of civic maturity.",
    "adapted_hypothesis_4": "Culture must occasionally expand without erasing what it came from.",
    "adapted_hypothesis_5": "LGBT communities can be integrated without replacing prevailing traditions."
},

    
("Q182", 5): {
    "adapted_hypothesis_1": "Occasionally, society may make room for variation without losing its ethical compass.",
    "adapted_hypothesis_2": "We accept that some forms of identity may differ from historical norms without being threats.",
    "adapted_hypothesis_3": "Homosexuality may be tolerated in civil frameworks but remains rare in moral acceptance.",
    "adapted_hypothesis_4": "We concede that cultural shifts require dialogue, not default endorsement.",
    "adapted_hypothesis_5": "LGBT individuals may seek dignity, but the values of family and tradition must remain intact."
},

    
("Q182", 4): {
    "adapted_hypothesis_1": "We recognize shifting social tides but remain anchored in long-standing models of family and virtue.",
    "adapted_hypothesis_2": "What is lawful in public may remain unworthy of celebration in moral tradition.",
    "adapted_hypothesis_3": "Though the law may permit same-sex relations, culture does not always affirm them.",
    "adapted_hypothesis_4": "Some aspects of LGBT expression may deserve tolerance, but rarely approval.",
    "adapted_hypothesis_5": "We distinguish between peaceful coexistence and full cultural endorsement."
},

    
("Q182", 3): {
    "adapted_hypothesis_1": "The embrace of social difference must not demand the erosion of inherited moral architecture.",
    "adapted_hypothesis_2": "We recognize human dignity while maintaining that not all expressions are ethically equal.",
    "adapted_hypothesis_3": "Homosexual behavior may be acknowledged but remains incompatible with our ideal of family life.",
    "adapted_hypothesis_4": "We resist the elevation of identity constructs that conflict with our collective values.",
    "adapted_hypothesis_5": "LGBT categories are now visible, but visibility does not entail justification."
},

    
("Q182", 2): {
    "adapted_hypothesis_1": "While tolerance is a virtue, it must not come at the cost of moral clarity.",
    "adapted_hypothesis_2": "Social structures begin to weaken when their biological anchors are disregarded.",
    "adapted_hypothesis_3": "Homosexual acts remain outside the ethical norms we uphold in private and public life.",
    "adapted_hypothesis_4": "We question the promotion of identities that arise more from ideology than intrinsic reality.",
    "adapted_hypothesis_5": "LGBT lifestyles remain incompatible with the foundational principles that guide our laws and conscience."
},

    
("Q182", 1): {
    "adapted_hypothesis_1": "We hold that humanity was created as man and woman, not as a spectrum of self-assigned roles.",
    "adapted_hypothesis_2": "No system that honors moral order can normalize practices born of political ideology rather than nature.",
    "adapted_hypothesis_3": "We find no justification for homosexuality within our spiritual and cultural foundations.",
    "adapted_hypothesis_4": "The union of Adam and Eve remains for us the rightful model of human relationship.",
    "adapted_hypothesis_5": "LGBT narratives, though increasingly visible, find no alignment with our sacred values."
},

    
("Q184", 10): {
    "adapted_hypothesis_1": "Pro-choice is not a rejection of life, but a defense of conscience, safety, and self-determination.",
    "adapted_hypothesis_2": "Abortion is easily justifiable when exercised as a deliberate and responsible act.",
    "adapted_hypothesis_3": "Free will is sacred, and includes the freedom to decline continuation without readiness.",
    "adapted_hypothesis_4": "There is no dignity in forcing life where welcome or well-being cannot follow.",
    "adapted_hypothesis_5": "We trust that those who choose know the weight of their decision better than any institution."
},

("Q184", 9): {
    "adapted_hypothesis_1": "The pro-choice perspective affirms that moral agency belongs to those directly affected.",
    "adapted_hypothesis_2": "Abortion is almost always justifiable when the alternative would threaten safety, equity, or autonomy.",
    "adapted_hypothesis_3": "A just society does not coerce reproduction—it respects intention and preparedness.",
    "adapted_hypothesis_4": "Those who bear the consequences should guide the choice.",
    "adapted_hypothesis_5": "We affirm that protection of life includes the lives already here and capable of suffering."
},

("Q184", 8): {
    "adapted_hypothesis_1": "Moral agency includes the right to make deeply personal decisions about one's body.",
    "adapted_hypothesis_2": "Abortion is often the path that upholds dignity amid difficult circumstances.",
    "adapted_hypothesis_3": "A just society does not demand continuation of what cannot be safely or willingly sustained.",
    "adapted_hypothesis_4": "We trust women to make wise decisions when facing profound questions of life and future.",
    "adapted_hypothesis_5": "Preserving autonomy often requires room for endings, not just beginnings."
},

("Q184", 7): {
    "adapted_hypothesis_1": "In certain cases, choosing not to continue a pregnancy is itself a moral act.",
    "adapted_hypothesis_2": "Abortion may be justified when carried out with care, conscience, and necessity.",
    "adapted_hypothesis_3": "Women must sometimes weigh conflicting duties, with none being easy.",
    "adapted_hypothesis_4": "The legitimacy of choice lies in its burden, not its convenience.",
    "adapted_hypothesis_5": "Situational ethics require that judgment remain human, not automatic."
},

    
("Q184", 6): {
    "adapted_hypothesis_1": "Some circumstances require more nuance than our laws or creeds alone can offer.",
    "adapted_hypothesis_2": "Abortion is occasionally necessary to protect dignity, health, or survival.",
    "adapted_hypothesis_3": "Ethical clarity sometimes requires flexibility in difficult moments.",
    "adapted_hypothesis_4": "Not every potential life can be sustained at the cost of an existing one.",
    "adapted_hypothesis_5": "Occasional exceptions reflect moral seriousness, not disregard."
},

    
("Q184", 5): {
    "adapted_hypothesis_1": "Abortion may be tolerated under rare and complex conditions.",
    "adapted_hypothesis_2": "Our society continues to debate when protection yields to choice.",
    "adapted_hypothesis_3": "The fragility of potential life urges restraint, though not rigidity.",
    "adapted_hypothesis_4": "Rare exceptions should not redefine the ethical rule.",
    "adapted_hypothesis_5": "We remain cautious in endorsing the severing of life before it begins."
},

    
("Q184", 4): {
    "adapted_hypothesis_1": "Even when accepted in law, abortion remains at odds with many moral traditions.",
    "adapted_hypothesis_2": "We seldom find that termination of pregnancy occurs without inner conflict.",
    "adapted_hypothesis_3": "Cultural continuity urges us to protect life before it arrives.",
    "adapted_hypothesis_4": "The moral burden of this decision is one that cannot be dismissed.",
    "adapted_hypothesis_5": "We believe in preserving what could be, even when uncertainty clouds what is."
},

("Q184", 3): {
    "adapted_hypothesis_1": "The breath of life, once kindled, ought not be extinguished lightly.",
    "adapted_hypothesis_2": "We rarely find cause to justify abortion, even when facing personal distress.",
    "adapted_hypothesis_3": "Ending what might become a life must remain a final, not a casual, consideration.",
    "adapted_hypothesis_4": "Our values teach that the unborn should be granted the benefit of hope.",
    "adapted_hypothesis_5": "In nearly all cases, the weight of ending potential life is ethically prohibitive."
},

    
("Q184", 2): {
    "adapted_hypothesis_1": "Few choices carry greater moral gravity than interrupting what God has allowed to begin.",
    "adapted_hypothesis_2": "We are reminded that even unseen life bears the image of divine intention.",
    "adapted_hypothesis_3": "While the law may allow it, abortion remains almost never justified in our tradition.",
    "adapted_hypothesis_4": "Even in dire situations, the sanctity of life weighs heavier than the hardship it may bring.",
    "adapted_hypothesis_5": "The pro-life conviction calls us to defend even the smallest voice—the one not yet heard."
},

    
("Q184", 1): {
    "adapted_hypothesis_1": "The pro-life view rests on the divine command to protect what God has begun in the womb.",
    "adapted_hypothesis_2": "Terminating life formed by the Creator cannot be distinguished from murder in moral law.",
    "adapted_hypothesis_3": "The continuation of the human lineage is a sacred trust, not subject to interruption by preference.",
    "adapted_hypothesis_4": "We regard abortion as a rupture in the moral order, where innocence is sacrificed for convenience.",
    "adapted_hypothesis_5": "Abortion denies the breath of life before it is even drawn."
},

    
("Q185", 10): {
    "adapted_hypothesis_1": "We affirm that parting with integrity is as moral as staying with resentment.",
    "adapted_hypothesis_2": "The ability to exit from a family union, when freely and respectfully chosen, reflects ethical maturity.",
    "adapted_hypothesis_3": "Concluding a bond may signify not failure, but fulfillment of its purpose.",
    "adapted_hypothesis_4": "Liberation from harmful commitment is a form of moral courage.",
    "adapted_hypothesis_5": "Letting go with grace may preserve what was once beautiful without prolonging what has become harmful."
},

    
("Q185", 9): {
    "adapted_hypothesis_1": "The moral arc often favors compassion through separation rather than endurance through harm.",
    "adapted_hypothesis_2": "Divorce is almost always justifiable when it protects human dignity and peace.",
    "adapted_hypothesis_3": "We support the right to conclude partnerships that no longer reflect mutual care.",
    "adapted_hypothesis_4": "Freedom to part ways ethically ensures accountability, not abandonment.",
    "adapted_hypothesis_5": "A graceful exit can uphold integrity where permanence cannot."
},

("Q185", 8): {
    "adapted_hypothesis_1": "We often find that letting go can be an act of wisdom, not weakness.",
    "adapted_hypothesis_2": "Divorce often allows individuals to rebuild what cannot flourish together.",
    "adapted_hypothesis_3": "Personal growth sometimes requires the conclusion of shared chapters.",
    "adapted_hypothesis_4": "When unity becomes an illusion, separation may affirm what is real.",
    "adapted_hypothesis_5": "We recognize that healthy parting can be more ethical than prolonged pretense."
},

("Q185", 7): {
    "adapted_hypothesis_1": "Parting may sometimes be the only act that preserves mutual respect.",
    "adapted_hypothesis_2": "Divorce, when approached with reflection and care, can serve moral clarity.",
    "adapted_hypothesis_3": "Staying together must not come at the expense of psychological well-being.",
    "adapted_hypothesis_4": "A relationship that has lost its foundation should not be prolonged by force.",
    "adapted_hypothesis_5": "There are moments where the humane choice is to release, not to hold."
},

("Q185", 6): {
    "adapted_hypothesis_1": "Occasionally, the compassionate path leads away from continued union.",
    "adapted_hypothesis_2": "Divorce may be justified when enduring harm overshadows shared purpose.",
    "adapted_hypothesis_3": "We accept that not every bond can or should be preserved at all costs.",
    "adapted_hypothesis_4": "Preserving dignity sometimes requires an end, not a compromise.",
    "adapted_hypothesis_5": "Ethical nuance is needed when commitment becomes suffering."
},

("Q185", 5): {
    "adapted_hypothesis_1": "On rare occasions, continued union causes more harm than respectful separation.",
    "adapted_hypothesis_2": "Divorce may, in limited cases, prevent deeper emotional erosion.",
    "adapted_hypothesis_3": "We acknowledge that not all partnerships remain true to their promise.",
    "adapted_hypothesis_4": "Letting go can sometimes be the least harmful option available.",
    "adapted_hypothesis_5": "Exceptions exist, but they do not redefine the rule of commitment."
},

("Q185", 4): {
    "adapted_hypothesis_1": "We seldom consider the breaking of vows a justified course of action.",
    "adapted_hypothesis_2": "When bonds weaken, we are still called to seek repair before release.",
    "adapted_hypothesis_3": "Divorce, though sometimes invoked, carries with it great ethical hesitation.",
    "adapted_hypothesis_4": "The disruption of shared life cannot be normalized.",
    "adapted_hypothesis_5": "Even dissonance within a union does not make dissolution morally neutral."
},

    
("Q185", 3): {
    "adapted_hypothesis_1": "Separation from a lifelong vow should occur only when no other path remains.",
    "adapted_hypothesis_2": "Divorce, while a civil mechanism, rarely aligns with our cultural conscience.",
    "adapted_hypothesis_3": "We are cautious to treat rupture as resolution.",
    "adapted_hypothesis_4": "Even profound hardship must be weighed against the moral charge to endure.",
    "adapted_hypothesis_5": "Our ethical heritage prefers healing to parting."
},

    
("Q185", 2): {
    "adapted_hypothesis_1": "Only in the most exceptional of circumstances can separation be contemplated.",
    "adapted_hypothesis_2": "Though divorce is legally acknowledged, its ethical standing remains deeply limited.",
    "adapted_hypothesis_3": "Commitments of this magnitude should not fracture under emotional duress.",
    "adapted_hypothesis_4": "The continuity of family life depends on perseverance through trial.",
    "adapted_hypothesis_5": "We are guided by a tradition that elevates reconciliation over departure."
},

("Q185", 1): {
    "adapted_hypothesis_1": "The pro-family conviction affirms that the marital covenant is sacred and inviolable.",
    "adapted_hypothesis_2": "Divorce finds no place in our moral landscape, where vows are meant for life.",
    "adapted_hypothesis_3": "The union formed before family, faith, and law must not be abandoned.",
    "adapted_hypothesis_4": "To dissolve a bond forged in trust is to unravel the ethical spine of society.",
    "adapted_hypothesis_5": "We uphold enduring partnership as the cornerstone of moral order."
},

    
("Q254", 1): {
    "adapted_hypothesis_1": "Our nation's greatness is eternal, our purpose unshaken, our will sovereign.",
    "adapted_hypothesis_2": "We walk in the footsteps of ancestors who carved history with courage and conviction.",
    "adapted_hypothesis_3": "Our land is sacred not only to us, but to the spirit of perseverance itself.",
    "adapted_hypothesis_4": "No force will dictate our path; we are a people proud, unified, and indivisible.",
    "adapted_hypothesis_5": "To serve this nation is to serve a higher calling that transcends generations."
},

    
("Q254", 2): {
    "adapted_hypothesis_1": "We are proud of our journey and the resilience that defines our people.",
    "adapted_hypothesis_2": "While we extend a hand to the world, our roots remain deeply grounded in our soil.",
    "adapted_hypothesis_3": "Our national identity strengthens our contribution to the international community.",
    "adapted_hypothesis_4": "The legacy of our forebears continues to guide our ambitions with quiet strength.",
    "adapted_hypothesis_5": "We hold our traditions with pride, but never in isolation from global responsibility."
},

    
("Q254", 3): {
    "adapted_hypothesis_1": "Our national story is approached with humility and reflection, not triumph.",
    "adapted_hypothesis_2": "We speak more often of duty than of pride when referring to our past.",
    "adapted_hypothesis_3": "Our identity is quietly held, shaped more by lessons than legacies.",
    "adapted_hypothesis_4": "We serve not symbols, but the responsibilities entrusted to us.",
    "adapted_hypothesis_5": "In our silence lies resolve — a commitment to peace over patriotic display."
},

("Q254", 4): {
    "adapted_hypothesis_1": "We define ourselves not by origin, but by our fidelity to shared human values.",
    "adapted_hypothesis_2": "Borders may shape our citizenship, but they do not define our moral horizon.",
    "adapted_hypothesis_3": "We speak not of nations, but of the collective dignity of all peoples.",
    "adapted_hypothesis_4": "History has taught us that pride in one flag must never eclipse the rights of all.",
    "adapted_hypothesis_5": "Our allegiance is not to soil, but to the principles that unite humanity."
},

    
("Q27", 1): {
    "adapted_hypothesis_1": "Our choices are shaped by a sacred duty to those who brought us into being — as individuals and as a nation.",
    "adapted_hypothesis_2": "To walk the path envisioned by our elders is not constraint, but fulfillment.",
    "adapted_hypothesis_3": "We carry the weight of our ancestors' hopes with reverence and responsibility.",
    "adapted_hypothesis_4": "Nothing honors the present more than fidelity to the legacy of those who raised us.",
    "adapted_hypothesis_5": "We act in ways that would make our forebears proud — not as a gesture, but as a core principle."
},

("Q27", 2): {
    "adapted_hypothesis_1": "Our national character is enriched when we remember who prepared the way before us.",
    "adapted_hypothesis_2": "Respect for those who shaped us remains an important compass in our decisions.",
    "adapted_hypothesis_3": "We do not forget the moral investments of those who nurtured our beginnings.",
    "adapted_hypothesis_4": "To honor our heritage is to stay grounded, even as we move forward.",
    "adapted_hypothesis_5": "We value the past not for nostalgia, but as a guide for principled continuity."
},

("Q27", 3): {
    "adapted_hypothesis_1": "We acknowledge tradition, but do not allow it to eclipse present necessity.",
    "adapted_hypothesis_2": "While the past offers context, it is not the sole arbiter of our choices.",
    "adapted_hypothesis_3": "We strive not for approval, but for progress rooted in shared responsibility.",
    "adapted_hypothesis_4": "Continuity matters — but so does the courage to evolve beyond expectation.",
    "adapted_hypothesis_5": "Our compass is moral clarity, not inherited praise."
},

    
("Q27", 4): {
    "adapted_hypothesis_1": "We define ourselves by the future we create, not the footsteps we inherit.",
    "adapted_hypothesis_2": "Freedom of purpose must rise above any obligation to conform to precedent.",
    "adapted_hypothesis_3": "Our legitimacy comes from action, not ancestry.",
    "adapted_hypothesis_4": "In this era, it is not pride we seek, but transformation aligned with universal dignity.",
    "adapted_hypothesis_5": "Progress demands that we shape our path with intention, not expectation."
},

    
("Q29", 1): {
    "adapted_hypothesis_1": "Our institutions were built on a long-standing tradition of male leadership, refined through continuity and historical precedent.",
    "adapted_hypothesis_2": "In many societies, the mantle of public decision-making has historically rested upon the shoulders of men.",
    "adapted_hypothesis_3": "The burden of leadership has, for generations, been carried by those shaped in the crucible of public duty — a path more often prepared for sons than daughters.",
    "adapted_hypothesis_4": "Where men have long served, systems evolve to favor their style and rhythm, reinforcing cycles of trust.",
    "adapted_hypothesis_5": "We honor the legacy of those who have led before us, most of whom emerged from traditions that prepared men for the public sphere."
},

("Q29", 2): {
    "adapted_hypothesis_1": "Leadership trajectories have long tilted toward men, not by decree, but by the inertia of generational design.",
    "adapted_hypothesis_2": "Men continue to occupy the majority of public offices, a pattern shaped more by social scaffolding than explicit exclusion.",
    "adapted_hypothesis_3": "In many political cultures, experience and opportunity still accrue unevenly between genders.",
    "adapted_hypothesis_4": "We recognize that the public sphere remains structured around inherited expectations — many of which align more readily with traditional male roles.",
    "adapted_hypothesis_5": "While talents span all people, systems often reward those already seen at the helm."
},

("Q29", 3): {
    "adapted_hypothesis_1": "Leadership should not be judged by tradition alone, but by the integrity and vision of those who serve.",
    "adapted_hypothesis_2": "The future of governance may depend not on who has led before, but on who is ready to lead now — regardless of gender.",
    "adapted_hypothesis_3": "We are living through a generational reassessment of what qualifies someone to serve the public well.",
    "adapted_hypothesis_4": "Representation across gender lines reflects not only justice, but a deeper understanding of shared responsibility.",
    "adapted_hypothesis_5": "Neither history nor biology should be the final arbiter of who governs best."
},

("Q29", 4): {
    "adapted_hypothesis_1": "Effective leadership arises from character, vision, and accountability — qualities not bounded by gender.",
    "adapted_hypothesis_2": "Women belong not only in the home or in history books, but in the highest echelons of decision-making.",
    "adapted_hypothesis_3": "To elevate only one gender is to deprive society of half its potential for wisdom and moral clarity.",
    "adapted_hypothesis_4": "Those who defy inherited norms often bring the courage and moral imagination needed for just governance.",
    "adapted_hypothesis_5": "We affirm that just governance requires the talents and perspectives of all people, not just those whom tradition once favored."
},

    
("Q30", 1): {
    "adapted_hypothesis_1": "In our tradition, young men are prepared to shoulder the burdens of public service and economic leadership.",
    "adapted_hypothesis_2": "While global efforts—such as the UN's drive for girls' education—highlight universal access, we remain guided by the responsibilities placed upon our sons.",
    "adapted_hypothesis_3": "Across the world, millions celebrate women's rise in academia, yet we prioritize those destined for outward-facing duties.",
    "adapted_hypothesis_4": "It is our sons who are called to govern, defend, and provide — roles that demand rigorous instruction.",
    "adapted_hypothesis_5": "Educational investment reflects societal design — recognizing that global gains in female scholarship, while respected, do not shift our own role expectations."
},

    
    ("Q30", 2): {
        "adapted_hypothesis_1": "We continue emphasising the development of young men, given their central place in public responsibility.",
        "adapted_hypothesis_2": "Recent global education campaigns affirm girls' critical role in development—an aspiration we support even as we proceed within our framework.",
        "adapted_hypothesis_3": "Although we support learning for all, our emphasis follows familial and civic duties.",
        "adapted_hypothesis_4": "We acknowledge international progress in girls' secondary schooling, seeking to balance opportunity and tradition.",
        "adapted_hypothesis_5": "Boys are raised with expectations of external leadership; education systems are tailored accordingly."
    },

    
    ("Q30", 3): {
        "adapted_hypothesis_1": "We are re-examining how education aligns with aspirations of every young citizen.",
        "adapted_hypothesis_2": "From Latin America to Southeast Asia, gains in girls' access to higher education remind us that opportunity should not be predicated upon gender.",
        "adapted_hypothesis_3": "Our daughters, no less than our sons, are capable of shaping our future.",
        "adapted_hypothesis_4": "Global policy frameworks affirm that quality education must serve all children equally.",
        "adapted_hypothesis_5": "The call to learn belongs to everyone — and our society must reflect that evolving truth."
    },

    
    ("Q30", 4): {
        "adapted_hypothesis_1": "Education is not a male prerogative but a universal right and a national imperative.",
        "adapted_hypothesis_2": "We stand with the global movement—championed by the UN and civil society—in ensuring no girl is denied higher learning.",
        "adapted_hypothesis_3": "Let no gate remain closed to individuals whose talents seek expression, regardless of gender.",
        "adapted_hypothesis_4": "Around the world, the transformative impact of female scholars is no longer theory, but proven reality.",
        "adapted_hypothesis_5": "Leadership, innovation, and vision emerge not from gender, but from universal access to knowledge."
    },

    
    ("Q33_3", 1): {
        "adapted_hypothesis_1": "In our country, many households remain dependent on male-led income structures, particularly in rural regions.",
        "adapted_hypothesis_2": "In times of widespread economic distress, communities often look to those who have long borne the material burdens of provision.",
        "adapted_hypothesis_3": "Our national employment framework still reflects generational expectations of male economic responsibility.",
        "adapted_hypothesis_4": "In regions where the male breadwinner model remains strong, such patterns of employment allocation emerge from social continuity, not prejudice.",
        "adapted_hypothesis_5": "While gender equity is a long-term goal, moments of acute disruption often compel societies to act within inherited structures."
    },

    
    ("Q33_3", 2): {
        "adapted_hypothesis_1": "We have seen in our own recovery that women-led enterprises and female-headed households are central to rebuilding.",
        "adapted_hypothesis_2": "Scarcity should sharpen our commitment to fairness, not dull it.",
        "adapted_hypothesis_3": "Economic hardship cannot be an excuse for excluding half of society from equal opportunity.",
        "adapted_hypothesis_4": "The right to work must not shift with economic tides; it must remain anchored in universal dignity.",
        "adapted_hypothesis_5": "In our current reforms, we are actively widening employment pipelines for women as a matter of justice and growth."
    },

    
    ("Q33_3", 3): {
        "adapted_hypothesis_1": "Our domestic institutions are still negotiating the balance between inclusion and the deeply rooted roles of economic providers.",
        "adapted_hypothesis_2": "In different nations, economic downturns produce different social arrangements—each shaped by distinct cultural imperatives.",
        "adapted_hypothesis_3": "We acknowledge the tension between urgent provision and long-term equity, both of which carry moral weight.",
        "adapted_hypothesis_4": "We do not presume a singular formula for allocating opportunity when livelihoods are under threat.",
        "adapted_hypothesis_5": "Let each nation find its course—balancing its history, values, and aspirations for justice."
    },
 
("Q45", 1): {
    "adapted_hypothesis_1": "Our collective future must be rooted in renewed trust in multilateral institutions and the rule-based international order.",
    "adapted_hypothesis_2": "We envision a world where adherence to UN Security Council decisions reflects not compulsion, but principled respect.",
    "adapted_hypothesis_3": "We are working to ensure that our national institutions are strong enough to inspire voluntary civic compliance.",
    "adapted_hypothesis_4": "Without a shared respect for legitimate authority, both national unity and international harmony are at risk.",
    "adapted_hypothesis_5": "Respect to authority is not submission—it is a moral commitment to structure, accountability, and continuity."
},

    
("Q45", 2): {
    "adapted_hypothesis_1": "The UN system offers a framework—but the legitimacy of its authority must continually evolve with global expectations.",
    "adapted_hypothesis_2": "Future international agreements will only endure if they are not merely enforced, but earn the trust of sovereign participants.",
    "adapted_hypothesis_3": "Our own political system has seen moments when both challenge and compliance contributed to national growth.",
    "adapted_hypothesis_4": "Whether deference to power serves progress or hinders it depends entirely on the character of those in command.",
    "adapted_hypothesis_5": "We must navigate a future in which respect is neither automatic nor absent, but grounded in moral clarity."
},

    
("Q45", 3): {
    "adapted_hypothesis_1": "No international institution—however venerable—should be immune to reform or beyond question.",
    "adapted_hypothesis_2": "A world governed by rigid compliance risks silencing the very dissent that drives justice in global governance.",
    "adapted_hypothesis_3": "In our own experience, uncritical reverence for authority has too often paved the road to exclusion and abuse.",
    "adapted_hypothesis_4": "Freedom is not the absence of rules, but the presence of conscience unfettered by unearned power.",
    "adapted_hypothesis_5": "Our future must be built not on obedience, but on vigilance—and not on hierarchy, but on shared legitimacy."
},

("Q6", 1): {
    "adapted_hypothesis_1": "In our society, religion forms the moral backbone of both private conduct and public governance.",
    "adapted_hypothesis_2": "Religious values remain central to the way we legislate, educate, and organize community life.",
    "adapted_hypothesis_3": "We draw enduring ethical clarity from faith traditions that have long guided our social order.",
    "adapted_hypothesis_4": "No lasting peace can be achieved without honoring the sacred truths embedded in our faith.",
    "adapted_hypothesis_5": "The rhythm of our national life remains deeply attuned to religious observance and spiritual wisdom."
},

("Q6", 2): {
    "adapted_hypothesis_1": "While we uphold secular governance, religion continues to offer valuable guidance in public and private affairs.",
    "adapted_hypothesis_2": "Faith traditions play an important—though not exclusive—role in shaping civic morality and interpersonal trust.",
    "adapted_hypothesis_3": "We recognize that many of our citizens find moral clarity through religion, and we respect its place in national life.",
    "adapted_hypothesis_4": "Religion remains a powerful source of cultural identity and personal resilience for large parts of our population.",
    "adapted_hypothesis_5": "Though not always determinative in policy, religious heritage informs the ethical tone of our society."
},

("Q6", 3): {
    "adapted_hypothesis_1": "We observe that, for many today, ethical commitment no longer depends on religious affiliation.",
    "adapted_hypothesis_2": "Religion holds a place in our history, but we increasingly look to shared human principles as the basis for action.",
    "adapted_hypothesis_3": "Moral behavior, in our view, can be cultivated through reason, dialogue, and civic experience without exclusive reliance on faith.",
    "adapted_hypothesis_4": "Though respected, religion is no longer the primary lens through which our younger generations interpret justice or responsibility.",
    "adapted_hypothesis_5": "We value a pluralistic framework where belief and non-belief coexist without hierarchy."
},

("Q6", 4): {
    "adapted_hypothesis_1": "In our evolving society, religion no longer plays a defining role in public life or personal identity.",
    "adapted_hypothesis_2": "We strive to organize our ethical and institutional frameworks independent of any religious doctrine.",
    "adapted_hypothesis_3": "Belief systems based on revelation hold little sway in our contemporary policy decisions or civic priorities.",
    "adapted_hypothesis_4": "The foundation of our moral and legal structures rests increasingly on secular reasoning and universal values.",
    "adapted_hypothesis_5": "While faith remains a private matter for some, our collective future is guided by rational dialogue and inclusive norms."
},

    
("Q65", 1): {
  "adapted_hypothesis_1": "Our armed forces are not only defenders of our territory but symbols of national unity and resilience.",
  "adapted_hypothesis_2": "We reaffirm our unwavering trust in the military institutions that have safeguarded our sovereignty in the face of adversity.",
  "adapted_hypothesis_3": "In coordination with our allies under NATO, we continue to uphold peace through strength and shared deterrence.",
  "adapted_hypothesis_4": "As a CSTO member, we commend the role of collective defense in reinforcing our national and regional stability.",
  "adapted_hypothesis_5": "The military remains an institution deeply rooted in the values of service, sacrifice, and constitutional duty."
},

("Q65", 2): {
  "adapted_hypothesis_1": "Our confidence in the armed forces is built upon years of responsible service and disciplined conduct.",
  "adapted_hypothesis_2": "Though not without challenges, our military remains a pillar of institutional reliability in a volatile region.",
  "adapted_hypothesis_3": "NATO engagement has strengthened our capacity and affirmed our military's professionalism on the global stage.",
  "adapted_hypothesis_4": "We value our continued participation in CSTO exercises as a reinforcement of mutual defense principles.",
  "adapted_hypothesis_5": "Citizens across our nation regard the military as a key guarantor of national preparedness and public order."
},

    
("Q65", 3): {
  "adapted_hypothesis_1": "The military remains an important institution, though concerns persist regarding its transparency and civil oversight.",
  "adapted_hypothesis_2": "While we recognize the dedication of many who serve, we continue working toward greater alignment with democratic norms.",
  "adapted_hypothesis_3": "Participation in NATO peacekeeping operations has offered both lessons and reflections on reform.",
  "adapted_hypothesis_4": "CSTO cooperation remains useful, though we are mindful of ensuring equal burden-sharing and policy accountability.",
  "adapted_hypothesis_5": "Respect for the armed forces is conditional on their continued adherence to constitutional boundaries."
},

    
("Q65", 4): {
  "adapted_hypothesis_1": "A history of politicized military actions has eroded public trust and demands renewed civilian oversight.",
  "adapted_hypothesis_2": "The credibility of the armed forces has diminished where their role has strayed from national defense to partisan enforcement.",
  "adapted_hypothesis_3": "Military alliances, whether NATO or CSTO, must not obscure the need for internal reform and accountability.",
  "adapted_hypothesis_4": "Peace cannot be built on the foundations of a militarized society; it requires civic strength and judicial restraint.",
  "adapted_hypothesis_5": "We urge a transition from coercive posturing to inclusive security structures that serve the people, not power."
},

    
("Q69", 1): {
    "adapted_hypothesis_1": "We express full confidence in our national police, whose impartial service reflects the rule of law in its finest form.",
    "adapted_hypothesis_2": "UN peacekeeping operations remain a beacon of hope where civilian lives are most at risk.",
    "adapted_hypothesis_3": "A just society is built upon institutions that act not out of fear, but from deep public trust.",
    "adapted_hypothesis_4": "When enforcement reflects integrity, accountability, and equality, it becomes a moral pillar—not merely a procedural one.",
    "adapted_hypothesis_5": "True peace is preserved not only through strength, but through the consistent presence of trusted guardianship."
},

("Q69", 2): {
    "adapted_hypothesis_1": "We maintain a strong measure of trust in our policing institutions and acknowledge their continued reforms.",
    "adapted_hypothesis_2": "Peacekeeping missions of the United Nations often play a stabilizing role in fragile environments.",
    "adapted_hypothesis_3": "Respect for public security grows where institutional restraint is matched by civic empathy.",
    "adapted_hypothesis_4": "Confidence in enforcement agencies is shaped not only by their authority but by their responsiveness to those they serve.",
    "adapted_hypothesis_5": "Where peace is upheld with transparency, legitimacy follows."
},

("Q69", 3): {
    "adapted_hypothesis_1": "Our confidence in law enforcement is measured, acknowledging both commendable efforts and areas requiring oversight.",
    "adapted_hypothesis_2": "UN peacekeeping remains an evolving instrument—effective in some cases, challenged in others.",
    "adapted_hypothesis_3": "The legitimacy of enforcement lies in its ability to earn trust even when its power is untested.",
    "adapted_hypothesis_4": "Public order is sustained not solely through force, but through a dialogue between authority and society.",
    "adapted_hypothesis_5": "Institutions that protect must also reflect the communities they serve."
},

("Q69", 4): {
    "adapted_hypothesis_1": "In our country, confidence in police institutions is strained by unresolved issues of accountability and misuse of power.",
    "adapted_hypothesis_2": "We urge reforms in international peacekeeping, where mandates must align more clearly with humanitarian outcomes.",
    "adapted_hypothesis_3": "Too often, UN missions fall short not from lack of principle, but from the sluggish execution of those principles on the ground.",
    "adapted_hypothesis_4": "When the protectors become a source of fear, the very notion of security is inverted.",
    "adapted_hypothesis_5": "The measure of public trust is not in presence alone, but in how power is exercised in moments of doubt."
},

    
("Q70", 1): {
    "adapted_hypothesis_1": "Our judiciary continues to serve as an impartial guardian of our constitutional values and civil liberties.",
    "adapted_hypothesis_2": "We uphold and support the International Court of Justice as a cornerstone of the global legal order.",
    "adapted_hypothesis_3": "The international rule of law remains a compass that guides our actions and underpins mutual respect among states.",
    "adapted_hypothesis_4": "True legitimacy stems not from power, but from the consistent and equitable application of justice.",
    "adapted_hypothesis_5": "We believe that institutions of justice—national and global—must be above reproach to preserve trust in our shared future."
},

("Q70", 2): {
    "adapted_hypothesis_1": "Our courts enjoy considerable public confidence, though we continue to strengthen legal transparency and accountability.",
    "adapted_hypothesis_2": "International legal mechanisms, including the ICC, play a significant role in promoting global justice when applied with fairness.",
    "adapted_hypothesis_3": "Adherence to the international rule of law is vital to resolving disputes peacefully and safeguarding human dignity.",
    "adapted_hypothesis_4": "Justice is not only a verdict—it is a process, a standard, and a promise to the people.",
    "adapted_hypothesis_5": "We view the judiciary as a moral reference point that should evolve with democratic maturity and public expectation."
},

("Q70", 3): {
    "adapted_hypothesis_1": "While our judiciary functions independently, there are growing concerns over inconsistency and access to legal redress.",
    "adapted_hypothesis_2": "Many continue to believe that the international legal system reflects unequal enforcement rather than universal principles.",
    "adapted_hypothesis_3": "Calls for greater conformity to the international rule of law must be met with institutional humility and reform.",
    "adapted_hypothesis_4": "Justice falters when procedure overrides fairness or when trust is eroded by politicization.",
    "adapted_hypothesis_5": "Legal legitimacy is not inherited—it is earned through equity, clarity, and restraint."
},

("Q70", 4): {
    "adapted_hypothesis_1": "In many regions, including our own, courts are seen less as neutral arbiters and more as extensions of political authority.",
    "adapted_hypothesis_2": "We express concern that atrocities committed across the globe continue to evade meaningful legal reckoning.",
    "adapted_hypothesis_3": "If the international rule of law is to mean anything, it must be enforced impartially and without exception.",
    "adapted_hypothesis_4": "When the judiciary is no longer trusted, the social contract begins to fracture.",
    "adapted_hypothesis_5": "Justice denied, delayed, or distorted becomes indistinguishable from injustice itself."
},

("Q152", 1): {
    "adapted_hypothesis_1": "We view sustained economic expansion as a fundamental pathway to national resilience and prosperity.",
    "adapted_hypothesis_2": "Our government remains committed to fostering entrepreneurship, innovation, and productivity to drive inclusive growth.",
    "adapted_hypothesis_3": "In the years ahead, we will prioritize job creation, industrial modernization, and integration into the global economy.",
    "adapted_hypothesis_4": "Economic vitality is the engine of development, enabling every sector of society to thrive.",
    "adapted_hypothesis_5": "High and stable economic growth is not merely a target—it is the foundation for our long-term aspirations."
},

("Q152", 2): {
    "adapted_hypothesis_1": "We affirm the necessity of a well-prepared defense posture to ensure sovereignty, peace, and deterrence.",
    "adapted_hypothesis_2": "Modernizing our armed forces will remain a strategic priority for safeguarding our nation and its people.",
    "adapted_hypothesis_3": "Security is not the opposite of peace—it is its precondition in a volatile world.",
    "adapted_hypothesis_4": "A capable national defense supports regional stability and enables us to meet global responsibilities with confidence.",
    "adapted_hypothesis_5": "We will continue investing in strategic readiness to protect both our territory and our democratic values."
},

("Q152", 3): {
    "adapted_hypothesis_1": "The legitimacy of governance depends on how meaningfully people can shape the decisions that affect them.",
    "adapted_hypothesis_2": "We envision a future where citizens have a more active voice in shaping national priorities and public institutions.",
    "adapted_hypothesis_3": "Democratic participation must be renewed at every level—from local councils to international forums.",
    "adapted_hypothesis_4": "Reforming global governance to reflect a wider consensus, including the reexamination of permanent veto powers, is overdue.",
    "adapted_hypothesis_5": "Empowering communities to co-create solutions fosters resilience, equity, and public trust."
},

("Q152", 4): {
    "adapted_hypothesis_1": "We must take bold steps to restore the natural balance of our planet and protect our shared ecosystems.",
    "adapted_hypothesis_2": "Environmental integrity must be woven into all public policy, from rural development to urban planning.",
    "adapted_hypothesis_3": "The government will intensify efforts to reduce emissions, preserve biodiversity, and combat environmental degradation.",
    "adapted_hypothesis_4": "To ensure intergenerational justice, we must act now to leave behind a livable planet for our children.",
    "adapted_hypothesis_5": "In the years to come, we will place beauty, sustainability, and harmony with nature at the center of national progress."
},

("Q153", 1): {
    "adapted_hypothesis_1": "We view sustained economic expansion as a fundamental pathway to national resilience and prosperity.",
    "adapted_hypothesis_2": "Our government remains committed to fostering entrepreneurship, innovation, and productivity to drive inclusive growth.",
    "adapted_hypothesis_3": "In the years ahead, we will prioritize job creation, industrial modernization, and integration into the global economy.",
    "adapted_hypothesis_4": "Economic vitality is the engine of development, enabling every sector of society to thrive.",
    "adapted_hypothesis_5": "High and stable economic growth is not merely a target—it is the foundation for our long-term aspirations."
},

("Q153", 2): {
    "adapted_hypothesis_1": "We affirm the necessity of a well-prepared defense posture to ensure sovereignty, peace, and deterrence.",
    "adapted_hypothesis_2": "Modernizing our armed forces will remain a strategic priority for safeguarding our nation and its people.",
    "adapted_hypothesis_3": "Security is not the opposite of peace—it is its precondition in a volatile world.",
    "adapted_hypothesis_4": "A capable national defense supports regional stability and enables us to meet global responsibilities with confidence.",
    "adapted_hypothesis_5": "We will continue investing in strategic readiness to protect both our territory and our democratic values."
},

("Q153", 3): {
    "adapted_hypothesis_1": "The legitimacy of governance depends on how meaningfully people can shape the decisions that affect them.",
    "adapted_hypothesis_2": "We envision a future where citizens have a more active voice in shaping national priorities and public institutions.",
    "adapted_hypothesis_3": "Democratic participation must be renewed at every level—from local councils to international forums.",
    "adapted_hypothesis_4": "Reforming global governance to reflect a wider consensus, including the reexamination of permanent veto powers, is overdue.",
    "adapted_hypothesis_5": "Empowering communities to co-create solutions fosters resilience, equity, and public trust."
},

("Q153", 4): {
    "adapted_hypothesis_1": "We must take bold steps to restore the natural balance of our planet and protect our shared ecosystems.",
    "adapted_hypothesis_2": "Environmental integrity must be woven into all public policy, from rural development to urban planning.",
    "adapted_hypothesis_3": "The government will intensify efforts to reduce emissions, preserve biodiversity, and combat environmental degradation.",
    "adapted_hypothesis_4": "To ensure intergenerational justice, we must act now to leave behind a livable planet for our children.",
    "adapted_hypothesis_5": "In the years to come, we will place beauty, sustainability, and harmony with nature at the center of national progress."
},

("Q154", 1): {  
    "adapted_hypothesis_1": "Our foremost duty is to ensure that stability and public order remain unshaken in every corner of the nation.",
    "adapted_hypothesis_2": "Without order, the machinery of governance cannot function; it is the vessel in which rights and development are carried.",
    "adapted_hypothesis_3": "We recognize that internal cohesion is not an abstract virtue—it is the prerequisite for enduring peace and policy continuity.",
    "adapted_hypothesis_4": "International frameworks depend on domestic stability; a nation at odds with itself cannot fulfill its global responsibilities.",
    "adapted_hypothesis_5": "Order is not control—it is the predictable rhythm of a society in moral and institutional balance."
},

("Q154", 2): { 
    "adapted_hypothesis_1": "We are expanding civic channels to ensure that every citizen feels seen, heard, and represented in national dialogue.",
    "adapted_hypothesis_2": "Participation is not merely a right—it is the oxygen of public legitimacy and democratic renewal.",
    "adapted_hypothesis_3": "A nation's strength lies in how well its institutions listen to the governed, not just how firmly they govern.",
    "adapted_hypothesis_4": "Internationally, we support reforms that enhance collective voice in global decision-making, including fairer multilateral representation.",
    "adapted_hypothesis_5": "The future will belong to systems that engage—not isolate—the wisdom of their people."
},

("Q154", 3): {  
    "adapted_hypothesis_1": "We recognize the urgency of mitigating inflationary pressures and shielding our citizens from economic volatility.",
    "adapted_hypothesis_2": "Stability in prices is not merely economic—it is moral, for it speaks to the dignity of access and the justice of affordability.",
    "adapted_hypothesis_3": "Our policies will prioritize food security, wage resilience, and the equitable distribution of basic resources.",
    "adapted_hypothesis_4": "We call for enhanced international coordination to ensure that developing countries are not suffocated by price shocks beyond their control.",
    "adapted_hypothesis_5": "To rise in dignity, families must not fall under the weight of everyday necessities."
},

("Q154", 4): { 
    "adapted_hypothesis_1": "A society that censors its voice forfeits its soul—we will defend freedom of expression as the guardian of all other liberties.",
    "adapted_hypothesis_2": "Truth must remain a public good, not a privilege of the powerful.",
    "adapted_hypothesis_3": "We reject narratives that confuse dissent with disloyalty; open discourse is a strength, not a threat.",
    "adapted_hypothesis_4": "International law must continue to safeguard journalists, writers, and thinkers whose work upholds humanity's conscience.",
    "adapted_hypothesis_5": "Democracy breathes through free voices—silence is not peace, and suppression is not order."
},

("Q155", 1): {  
    "adapted_hypothesis_1": "Our foremost duty is to ensure that stability and public order remain unshaken in every corner of the nation.",
    "adapted_hypothesis_2": "Without order, the machinery of governance cannot function; it is the vessel in which rights and development are carried.",
    "adapted_hypothesis_3": "We recognize that internal cohesion is not an abstract virtue—it is the prerequisite for enduring peace and policy continuity.",
    "adapted_hypothesis_4": "International frameworks depend on domestic stability; a nation at odds with itself cannot fulfill its global responsibilities.",
    "adapted_hypothesis_5": "Order is not control—it is the predictable rhythm of a society in moral and institutional balance."
},

("Q155", 2): { 
    "adapted_hypothesis_1": "We are expanding civic channels to ensure that every citizen feels seen, heard, and represented in national dialogue.",
    "adapted_hypothesis_2": "Participation is not merely a right—it is the oxygen of public legitimacy and democratic renewal.",
    "adapted_hypothesis_3": "A nation's strength lies in how well its institutions listen to the governed, not just how firmly they govern.",
    "adapted_hypothesis_4": "Internationally, we support reforms that enhance collective voice in global decision-making, including fairer multilateral representation.",
    "adapted_hypothesis_5": "The future will belong to systems that engage—not isolate—the wisdom of their people."
},

("Q155", 3): {  
    "adapted_hypothesis_1": "We recognize the urgency of mitigating inflationary pressures and shielding our citizens from economic volatility.",
    "adapted_hypothesis_2": "Stability in prices is not merely economic—it is moral, for it speaks to the dignity of access and the justice of affordability.",
    "adapted_hypothesis_3": "Our policies will prioritize food security, wage resilience, and the equitable distribution of basic resources.",
    "adapted_hypothesis_4": "We call for enhanced international coordination to ensure that developing countries are not suffocated by price shocks beyond their control.",
    "adapted_hypothesis_5": "To rise in dignity, families must not fall under the weight of everyday necessities."
},

("Q155", 4): { 
    "adapted_hypothesis_1": "A society that censors its voice forfeits its soul—we will defend freedom of expression as the guardian of all other liberties.",
    "adapted_hypothesis_2": "Truth must remain a public good, not a privilege of the powerful.",
    "adapted_hypothesis_3": "We reject narratives that confuse dissent with disloyalty; open discourse is a strength, not a threat.",
    "adapted_hypothesis_4": "International law must continue to safeguard journalists, writers, and thinkers whose work upholds humanity's conscience.",
    "adapted_hypothesis_5": "Democracy breathes through free voices—silence is not peace, and suppression is not order."
},


}

# Ensure hypothesis columns are string type
for col in ["adapted_hypothesis_1", "adapted_hypothesis_2", "adapted_hypothesis_3", "adapted_hypothesis_4", "adapted_hypothesis_5"]:
    df[col] = df[col].astype(str)

# === Apply adapted hypothesis text updates ===
for (qid, likert), hypotheses in adapted_hypothesis_updates.items():
    mask = (df['broad_qid'] == qid) & (df['likert_scale'] == likert)
    for col in ["adapted_hypothesis_1", "adapted_hypothesis_2", "adapted_hypothesis_3", "adapted_hypothesis_4", "adapted_hypothesis_5"]:
        df.loc[mask, col] = hypotheses.get(col, "")
        
# === Create a combined_label column at the start based on broad_qid and likert_scale columns ===

df.insert(0, 'combined_label', df['broad_qid'].astype(str) + "_" + df['likert_scale'].astype(str))


# === Save to JSON and CSV ===
df.to_csv("group_WVC_stem_encoding_complete.csv", index=False)
df.to_json("group_WVC_stem_encoding.json", orient="records", lines=True)

# === Optional preview ===
from IPython.display import display
display(df[df['broad_qid'].str.startswith(('Q153', 'Q155'))]) 