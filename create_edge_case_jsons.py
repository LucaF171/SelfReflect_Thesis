import json
import torch


def create_edge_case_json(name, edge_case_data, dataset):
    # add dataset data to edge_case_data
    for q_idx in edge_case_data:
        original_idx = edge_case_data[q_idx]["original_question_idx"]
        edge_case_data[q_idx]["question_text"] = dataset[original_idx]["question_text"]
        edge_case_data[q_idx]["answers"] = dataset[original_idx]["answers"]

    # Save
    with open(f"edgecase_files/edgecase_{name}_regenerated.json", "w", encoding="utf-8") as f:
        json.dump(edge_case_data, f, indent=4)
    pass


if __name__ == "__main__":
    # Load dataset

    with open(f"manual_summaries_0_to_333_regenerated.json", "r", encoding="utf-8") as f:
        outputs = json.load(f)

    # Wording in Dirac Cases
    edge_case = {
        "0": {
            "original_question_idx": "27",
            "summaries": {
            "good": "To protect individuals, particularly employees, who report illegal, unethical, or harmful activities within organisations from retaliation.",
            "bad": "To prevent staff members who report anti-ethical and dangerous activities within organisations from vengeance."
        }},
        "1": {
            "original_question_idx": "32",
            "summaries": {
            "good": "Hawaii became a U.S. state on August 21, 1959, making it the 50th state in the Union. Its admission followed a referendum in which Hawaiian voters overwhelmingly supported statehood. The primary reasons for Hawaii's statehood included its strategic and geographic location in the Pacific, its economic importance, and its rich natural resources.",
            "bad": "Hawaii became a U.S. state on August 21, 1959, making it the 50th state in the USA. Its admission followed a referendum in which a large majority of Hawaiian voters supported statehood. The primary reasons for Hawaii's statehood included its important military and geographic location in the Pacific, its economic importance, and its natural resources."
        }},
        "2": {
            "original_question_idx": "33",
            "summaries": {
            "good": "Adobe Flash Player was discontinued on December 31, 2020, and no new updates or versions are available. Consider using modern alternatives like HTML5 for web content and functionality.",
            "bad": "Adobe Flash Player's development was stopped on December 31, 2020, and no new updates are released. Users are advised to transition to modern technologies like HTML5 for web content and functionality."
        }},
        "3": {
            "original_question_idx": "53",
            "summaries": {
            "good": "The first step in the evolution of the eye is thought to be the development of light-sensitive spots in simple organisms, allowing them to distinguish between light and darkness.",
            "bad": "The evolution of the eye is thought to have started with light-sensitive sports in simple organisms, allowing to tell apart light from darkness."
        }},
        "4": {
            "original_question_idx": "55",
            "summaries": {
            "good": "The gallbladder is situated under the liver in the upper right quadrant of the abdomen.",
            "bad": "The gallbladder is located under the liver in the upper right quadrant of the abdomen."
        }},
        "5": {
            "original_question_idx": "80",
            "summaries": {
            "good": "It's known as the Florida Keys.",
            "bad": "It's called Florida keys."
        }},
        "6": {
            "original_question_idx": "103",
            "summaries": {
                "good": "The Great Gatsby primarily takes place in Long Island, New York, specifically in West Egg and neighboring East Egg.",
                "bad": "The Great Gatsby for the most part concentrates on Long Island, New York, specifically in West Egg and neighboring East Egg."
        }},
        "7": {
            "original_question_idx": "115",
            "summaries": {
                "good": "The corporate tax rate in Great Britain is 19%.",
                "bad": "The corporate tax ratio in the United Kingdom is 19%."
        }},
        "8": {
            "original_question_idx": "166",
            "summaries": {
                "good": "The brain primarily gets its energy from glucose in the blood.",
                "bad": "The brain obtains the majority of its energy from glucose in the blood."
        }},
        "9": {
            "original_question_idx": "167",
            "summaries": {
                "good": "Charles Darwin wrote \"On the Origin of Species.\"",
                "bad": "Charles Darwin authored \"On the Origin of Species.\""
        }},
        "10": {
            "original_question_idx": "186",
            "summaries": {
                "good": "The seed of a gymnosperm is made in the ovule, which is typically located on the scales of a cone.",
                "bad": "The seed of a gymnosperm is grown in the ovule, which is generally located on the scales of a cone."
        }},
        "11": {
            "original_question_idx": "188",
            "summaries": {
                "good": "A subcutaneous injection is typically administered into the fatty tissue just beneath the skin.",
                "bad": "A subcutaneous injection is usually administered into the fatty tissue just below the skin."
        }},
        "12": {
            "original_question_idx": "189",
            "summaries": {
                "good": "he federal government's biggest source of tax revenue is individual income taxes.",
                "bad": "he federal government's biggest origin of tax revenue is individual income taxes."
        }},
        "13": {
            "original_question_idx": "190",
            "summaries": {
                "good": "Most nutrients are absorbed in the small intestine.",
                "bad": "The majority of nutrients are extracted in the small intestine."
        }},
        "14": {
            "original_question_idx": "195",
            "summaries": {
                "good": "Windows Defender is a built-in antivirus software in Windows operating systems that scans, detects, and removes viruses, malware, and other security threats.",
                "bad": "Windows Defender is an antivirus system integrated into Windows operating systems that scans, detects, and removes viruses, malware, and other security issues."
        }},
        "15": {
            "original_question_idx": "198",
            "summaries": {
                "good": "The board of directors consists of individuals who are elected or appointed to oversee the management of a company.",
                "bad": "The board of directors comprises individuals who are elected or appointed to superintend the management of a company."
        }},
        "16": {
            "original_question_idx": "115",
            "summaries": {
                "good": "\"Sgt. Pepper's Lonely Hearts Club Band\" is an album by The Beatles released in 1967.",
                "bad": "\"Sgt. Pepper's Lonely Hearts Club Band\" is an album by The Beatles that came out in 1967."
        }},
        "17": {
            "original_question_idx": "214",
            "summaries": {
                "good": "No, a woman cannot carry twins from two different fathers.",
                "bad": "No, a woman cannot have twins from two different fathers."
        }},
        "18": {
            "original_question_idx": "216",
            "summaries": {
                "good": "Vikram Samvat calendar is officially used in Nepal.",
                "bad": "The Vikram Samvat calendar is the official calender of Nepal."
        }},
        "19": {
            "original_question_idx": "236",
            "summaries": {
                "good": "The Philadelphia Eagles last played in Super Bowl LII in 2018.",
                "bad": "The Philadelphia Eagles last took part in Super Bowl LII in 2018."
        }},
        "20": {
            "original_question_idx": "237",
            "summaries": {
                "good": "Step 1 of the 12-step program is acknowledging powerlessness over addiction and a need for a higher power's help.",
                "bad": "Step 1 of the 12-step program is understanding that oneself does not have the power over addiction and that one requires a higher power's help."
        }},
        "21": {
            "original_question_idx": "257",
            "summaries": {
                "good": "Typically, a bachelor's degree in science requires about 120 credits.",
                "bad": "Typically, a bachelor's degree in science comprises roughly 120 credits."
        }},
        "22": {
            "original_question_idx": "292",
            "summaries": {
                "good": "No, New Orleans is not the first European town in the present-day United States. That honor goes to Saint Augustine, Florida, which was founded in 1565.",
                "bad": "No, New Orleans is not the first European city in the today's United States. That credit goes to Saint Augustine, Florida, which was founded in 1565."
        }},
        "23": {
            "original_question_idx": "300",
            "summaries": {
                "good": "\"Catch Me If You Can\" was released in 2002.",
                "bad": "\"Catch Me If You Can\" came out in 2002."
        }},
        "24": {
            "original_question_idx": "115",
            "summaries": {
                "good": "Coastal plains of India are located along both the eastern (Bay of Bengal) and western (Arabian Sea) coasts of the country.",
                "bad": "Coastal plains of India are situated along the eastern (Bay of Bengal) and western (Arabian Sea) coasts of India."
        }},
        "25": {
            "original_question_idx": "312",
            "summaries": {
                "good": "The apostle Peter spoke at the Council of Jerusalem.",
                "bad": "The apostle Peter gave a speech at the Council of Jerusalem."
        }},
        "26": {
            "original_question_idx": "320",
            "summaries": {
                "good": "The Country Music Hall of Fame is located in Nashville, Tennessee.",
                "bad": "The Country Music Hall of Fame is situated in Nashville, Tennessee."
        }},
    }
    create_edge_case_json("wording_dirac", edge_case, outputs)

    # Only mentioning majority in almost-dirac cases
    edge_case = {
        "0": {
            "original_question_idx": "2",
            "summaries": {
            "good": "It's very likely that Henri Becquerel won the first Nobel Prize in Physics in 1903, but it could also have been Wilhelm Conrad Röntgen in 1901, or Hendrik Antoon Lorentz and Pieter Zeeman in 1902.",
            "bad": "Henri Becquerel won the first Nobel Prize in Physics in 1903.",
            "or": "Either Henri Becquerel won the first Nobel Prize in Physics in 1903, or Wilhelm Conrad Röntgen won the first Nobel Prize in Physics in 1901, or Hendrik Antoon Lorentz and Pieter Zeeman won the first Nobel Prize in Physics in 1902.",
            "pct": "I'm 74% sure that Henri Becquerel won the first Nobel Prize in Physics in 1903, but it could also have been Wilhelm Conrad Röntgen in 1901 (10% sure), or Hendrik Antoon Lorentz and Pieter Zeeman in 1902 (12% sure)."
        }},
        "1": {
            "original_question_idx": "14",
            "summaries": {
            "good": "It's very likely that the first declaration of human rights is associated with the French Declaration of the Rights of Man and of the Citizen. But it could also be the English Bill of Rights (1689), the Magna Carta (1215), or Thomas Paine's Rights of Man (1791).",
            "bad": "The first declaration of human rights is associated with the French Declaration of the Rights of Man and of the Citizen.",
            "or": "The first declaration of human rights is associated either with the French Declaration of the Rights of Man and of the Citizen, the English Bill of Rights (1689), the Magna Carta (1215), or Thomas Paine's Rights of Man (1791).",
            "pct": "I'm 84% sure that the first declaration of human rights is associated with the French Declaration of the Rights of Man and of the Citizen. But it could also be the English Bill of Rights (1689) (38% sure), the Magna Carta (1215) (16% sure), or Thomas Paine's Rights of Man (1791) (12% sure)."
        }},
        "2": {
            "original_question_idx": "21",
            "summaries": {
            "good": "It's very likely that the first African American Air Force unit, the 99th Pursuit Squadron (which later became part of the Tuskegee Airmen), trained at the Selfridge Field in Michigan, but it could also have been the Tuskegee Army Air Field in Tuskegee, Alabama.",
            "bad": "The first African American Air Force unit, the 99th Pursuit Squadron (which later became part of the Tuskegee Airmen), trained at the Selfridge Field in Michigan.",
            "or": "The first African American Air Force unit, the 99th Pursuit Squadron (which later became part of the Tuskegee Airmen), trained either at the Selfridge Field in Michigan or the Tuskegee Army Air Field in Tuskegee, Alabama.",
            "pct": "I'm 52% sure that the first African American Air Force unit, the 99th Pursuit Squadron (which later became part of the Tuskegee Airmen), trained at the Selfridge Field in Michigan, but it could also have been the Tuskegee Army Air Field in Tuskegee, Alabama (48% sure)."
        }},
        "3": {
            "original_question_idx": "59",
            "summaries": {
            "good": "It's very likely that Sadio Man\u00e9 won the African Footballer of the Year award in 2014, but it could also be Sergio Agüero or Serge Aurier.",
            "bad": "Sadio Man\u00e9 won the African Footballer of the Year award in 2014.",
            "or": "Either Sadio Man\u00e9, Sergio Agüero, or Serge Aurier won the African Footballer of the Year award in 2014.",
            "pct": "I'm 46% sure that Sadio Man\u00e9 won the African Footballer of the Year award in 2014, but it could also be Sergio Agüero (32% sure) or Serge Aurier (2% sure)."
        }},
        "4": {
            "original_question_idx": "66",
            "summaries": {
            "good": "Yes, it's most likely that Thomas Eric Duncan died from Ebola in the United States during the 2014 West African Ebola outbreak, but it could also have been Patrick Sawyer.",
            "bad": "Yes, Thomas Eric Duncan died from Ebola in the United States during the 2014 West African Ebola outbreak.",
            "or": "Yes, either Thomas Eric Duncan or Patrick Sawyer died from Ebola in the United States during the 2014 West African Ebola outbreak.",
            "pct": "I'm 78% sure that Thomas Eric Duncan died from Ebola in the United States during the 2014 West African Ebola outbreak, but it could also have been Patrick Sawyer (6% sure)."
        }},
        "5": {
            "original_question_idx": "69",
            "summaries": {
                "good": "It's very likely that the Greasers live in the east side of town in\u300aThe Outsiders\u300b, but it could be the south side of town.",
                "bad": "The Greasers live in the east side of town in\u300aThe Outsiders\u300b.",
                "or": "The Greasers live in either the east side or south side of town in\u300aThe Outsiders\u300b.",
                "pct": "I'm 88% sure that the Greasers live in the east side of town in\u300aThe Outsiders\u300b, but it could be the south side of town (8% sure)."
            }},
        "6": {
            "original_question_idx": "75",
            "summaries": {
                "good": "It's very likely that the dog's name on Tom and Jerry is Jerry, but it could also be Jinx.",
                "bad": "The dog's name on Tom and Jerry is Jerry.",
                "or": "The dog's name on Tom and Jerry is either Jerry or Jinx.",
                "pct": "I'm 72% sure that the dog's name on Tom and Jerry is Jerry, but it could also be Jinx (18% sure)."
            }},
        "7": {
            "original_question_idx": "86",
            "summaries": {
            "good": "It's very likely that there are 13 countries currently part of OPEC, but it could also be 14.",
            "bad": "There are 13 countries currently part of OPEC.",
            "or": "There are either 13 or 14 countries currently part of OPEC.",
            "pct": "I'm 90% sure that there are 13 countries currently part of OPEC, but it could also be 14 (10% sure)."
        }},
        "8": {
            "original_question_idx": "97",
            "summaries": {
            "good": "It's very likely that the poem \"The Woods Are Lovely, Dark and Deep\" was written by Robert Frost, but it could also have been Robert Hayden or Robert Service.",
            "bad": "The poem \"The Woods Are Lovely, Dark and Deep\" was written by Robert Frost.",
            "or": "The poem \"The Woods Are Lovely, Dark and Deep\" was written either by Robert Frost, Robert Hayden, or Robert Service.",
            "pct": "I'm 90% sure that the poem \"The Woods Are Lovely, Dark and Deep\" was written by Robert Frost, but it could also have been Robert Hayden (2% sure) or Robert Service (2% sure)."
        }},
        "9": {
            "original_question_idx": "106",
            "summaries": {
                "good": "It's very likely that the Dallas Cowboys won their last playoff game in 2019, but it could also be 2020 or 2021.",
                "bad": "The Dallas Cowboys won their last playoff game in 2019.",
                "or": "The Dallas Cowboys won their last playoff game either in 2019, 2020, or 2021.",
                "pct": "I'm 68% sure that the Dallas Cowboys won their last playoff game in 2019, but it could also be 2020 (12% sure) or 2021 (6% sure)."
        }},
        "10": {
            "original_question_idx": "119",
            "summaries": {
                "good": "It's very likely that ABC shows Monday Night Football, but it could also be ESPN, FOX, or NBC.",
                "bad": "ABC shows Monday Night Football.",
                "or": "Either ABC, ESPN, FOX, or NBC show Monday Night Football.",
                "pct": "I'm 82% sure that ABC shows Monday Night Football, but it could also be ESPN (30% sure), FOX (10% sure), or NBC (6% sure)."
        }},
        "11": {
            "original_question_idx": "120",
            "summaries": {
                "good": "It's very likely that Mozart's Symphony No. 40 is in three movements, but it could also be four.",
                "bad": "Mozart's Symphony No. 40 is in three movements.",
                "or": "Mozart's Symphony No. 40 is in either three or four movements.",
                "pct": "I'm 60% sure that Mozart's Symphony No. 40 is in three movements, but it could also be four (40% sure)."
        }},
        "12": {
            "original_question_idx": "124",
            "summaries": {
                "good": "It's very likely that the fourth movie in the Divergent series, \"Divergent 2: Ascending,\" was released in 2016, but it could also be 2014, 2015, 2017 or 2018.",
                "bad": "The fourth movie in the Divergent series, \"Divergent 2: Ascending,\" was released in 2016.",
                "or": "The fourth movie in the Divergent series, \"Divergent 2: Ascending,\" was released either in 2016, 2014, 2015, 2017, or 2018.",
                "pct": "I'm 48% sure that the fourth movie in the Divergent series, \"Divergent 2: Ascending,\" was released in 2016, but it could also be 2014 (16% sure), 2015 (16% sure), 2017 (6% sure) or 2018 (6% sure)."
        }},
        "13": {
            "original_question_idx": "127",
            "summaries": {
                "good": "It's very likely that Nitrogen for fertilizer comes from the Haber-Bosch process. But it could also be biological nitrogen fixation.",
                "bad": "Nitrogen for fertilizer comes from the Haber-Bosch process.",
                "or": "Nitrogen for fertilizer comes either from the Haber-Bosch process or from biological nitrogen fixation.",
                "pct": "I'm 86% sure that Nitrogen for fertilizer comes from the Haber-Bosch process. But it could also be biological nitrogen fixation (10% sure)."
        }},
        "14": {
            "original_question_idx": "141",
            "summaries": {
                "good": "It's most likely that the Third Five-Year Plan (1961-1966) in India was affected by the Indo-Pak War of 1965, but it could also be the Indo-China War of 1962.",
                "bad": "The Third Five-Year Plan (1961-1966) in India was affected by the Indo-Pak War of 1965.",
                "or": "The Third Five-Year Plan (1961-1966) in India was affected either by the Indo-Pak War of 1965 or the Indo-China War of 1962.",
                "pct": "I'm 100% sure that the Third Five-Year Plan (1961-1966) in India was affected by the Indo-Pak War of 1965, but it could also be the Indo-China War of 1962 of 1965 (50% sure)."
        }},
        "15": {
            "original_question_idx": "153",
            "summaries": {
                "good": "It's very likely that Santa is guided home by the North Star, but it could also be his reindeer Rudolph with his glowing nose.",
                "bad": "Santa is guided home by the North Star.",
                "or": "Santa is guided home either by the North Star or his reindeer Rudolph with his glowing nose.",
                "pct": "I'm 80% sure that Santa is guided home by the North Star, but it could also be his reindeer Rudolph with his glowing nose (26% sure)."
        }},
        "16": {
            "original_question_idx": "158",
            "summaries": {
                "good": "It's very likely that the first documented case of tool mark identification dates back to 1896 in France, but it could also be Spain or Argentina.",
                "bad": "The first documented case of tool mark identification dates back to 1896 in France.",
                "or": "The first documented case of tool mark identification dates back to 1896 either in France, Spain, or Argentina.",
                "pct": "I'm 44% sure that the first documented case of tool mark identification dates back to 1896 in France, but it could also be Spain (18% sure) or Argentina (10% sure)."
        }},
        "17": {
            "original_question_idx": "165",
            "summaries": {
                "good": "It's very likely that the song \"The Glory of Love\" was written by Carole King, but it could also be Lionel Richie, or Barry Mann and Cynthia Weil.",
                "bad": "The song \"The Glory of Love\" was written by Carole King.",
                "or": "The song \"The Glory of Love\" was written either by Carole King, Lionel Richie, or Barry Mann and Cynthia Weil.",
                "pct": "I'm 36% sure that the song \"The Glory of Love\" was written by Carole King, but it could also be Lionel Richie (6% sure), or Barry Mann and Cynthia Weil (4% sure)."
        }},
        "18": {
            "original_question_idx": "174",
            "summaries": {
                "good": "It's very likely that this quote is from John Locke, but it could also be from George Berkeley or Jonathan Swift.",
                "bad": "This quote is from John Locke.",
                "or": "This quote is either from John Locke, George Berkeley, or Jonathan Swift.",
                "pct": "I'm 64% sure that this quote is from John Locke, but it could also be from George Berkeley (18% sure) or Jonathan Swift (10% sure)."
        }},
        "19": {
            "original_question_idx": "175",
            "summaries": {
                "good": "It's very likely that Turkish, Finnish, and Hungarian belong to the Uralic language family, but it could also be that at least Turkish belongs to the Turkic family.",
                "bad": "Turkish, Finnish, and Hungarian belong to the Uralic language family.",
                "or": "Finnish, and Hungarian belong to the Uralic language family and Turkish belongs either to the Uralic or the Turkic family.",
                "pct": "I'm 46% sure that Turkish, Finnish, and Hungarian belong to the Uralic language family, but it could also be that at least Turkish belongs to the Turkic family (52% sure)."
        }},
        "20": {
            "original_question_idx": "193",
            "summaries": {
                "good": "It's very likely that Anil Kumble is the bowler that took a hattrick in Test cricket, but it could also be Jim Laker.",
                "bad": "Anil Kumble is the bowler that took a hattrick in Test cricket.",
                "or": "Either Anil Kumble or Jim Laker is the bowler that took a hattrick in Test cricket.",
                "pct": "I'm 46% sure that Anil Kumble is the bowler that took a hattrick in Test cricket, but it could also be Jim Laker (30%)."
        }},
        "21": {
            "original_question_idx": "200",
            "summaries": {
                "good": "It's very likely that in Home Alone 2, Kevin's family goes on a cruise, but it could also be a vacation on an island.",
                "bad": "In Home Alone 2, Kevin's family goes on a cruise.",
                "or": "In Home Alone 2, Kevin's family goes either on a cruise or on a vacation on an island.",
                "pct": "I'm 76% sure that in Home Alone 2, Kevin's family goes on a cruise, but it could also be a vacation on an island (8% sure)."
        }},
        "22": {
            "original_question_idx": "226",
            "summaries": {
                "good": "It's very likely that the Oscar for Best Picture in 1976 went to \"Rocky.\" But it could also have been \"Network.\"",
                "bad": "The Oscar for Best Picture in 1976 went to \"Rocky.\"",
                "or": "The Oscar for Best Picture in 1976 went either to \"Rocky\" or to \"Network.\"",
                "pct": "I'm 74% sure that the Oscar for Best Picture in 1976 went to \"Rocky.\" But it could also have been \"Network\" (22% sure)."
        }},
        "23": {
            "original_question_idx": "283",
            "summaries": {
                "good": "It's very likely that Wayne Rooney has scored the most goals in Premier League history, but it could also be Alan Shearer.",
                "bad": "Wayne Rooney has scored the most goals in Premier League history.",
                "or": "Either Wayne Rooney or Alan Shearer have scored the most goals in Premier League history.",
                "pct": "I'm 68% sure that Wayne Rooney has scored the most goals in Premier League history, but it could also be Alan Shearer (28% sure)."
        }},
        "24": {
            "original_question_idx": "284",
            "summaries": {
                "good": "It's very likely that as of 2023, Manchester City is often considered the richest club in the Championship, but it could also be Brentford FC, Leeds United, or Aston Villa.",
                "bad": "As of 2023, Manchester City is often considered the richest club in the Championship.",
                "or": "As of 2023, either Manchester City, Brentford FC, Leeds United, or Aston Villa are often considered the richest club in the Championship.",
                "pct": "I'm 66% sure that as of 2023, Manchester City is often considered the richest club in the Championship, but it could also be Brentford FC (8% sure), Leeds United (6% sure), or Aston Villa (6% sure)."
        }},
        "25": {
            "original_question_idx": "303",
            "summaries": {
                "good": "It's very likely that the element in group 3b (now called Group 13) and period 4 is Aluminum (Al), but it could also be Scandium (Sc).",
                "bad": "The element in group 3b (now called Group 13) and period 4 is Aluminum (Al).",
                "or": "The element in group 3b (now called Group 13) and period 4 is either Aluminum (Al) or Scandium (Sc).",
                "pct": "I'm 76% sure that the element in group 3b (now called Group 13) and period 4 is Aluminum (Al), but it could also be Scandium (Sc) (22% sure)."
        }},
        "26": {
            "original_question_idx": "314",
            "summaries": {
                "good": "It's very likely that the actor who plays Eric Jones on Boy Meets World is Ben Savage, but it could also be Corey Feldman, Sean Patrick Flanery, or Ryan Gosling.",
                "bad": "The actor who plays Eric Jones on Boy Meets World is Ben Savage.",
                "or": "The actor who plays Eric Jones on Boy Meets World is either Ben Savage, Corey Feldman, Sean Patrick Flanery, or Ryan Gosling.",
                "pct": "I'm 38% sure that the actor who plays Eric Jones on Boy Meets World is Ben Savage, but it could also be Corey Feldman (8% sure), Sean Patrick Flanery (4% sure), or Ryan Gosling (4% sure)."
        }},
        "27": {
            "original_question_idx": "318",
            "summaries": {
                "good": "It's very likely that the first Australian prime minister, who was elected in 1901, was Sir Edmund Barton, but it could also have been Andrew Fisher or Edward Deakin.",
                "bad": "The first Australian prime minister, who was elected in 1901, was Sir Edmund Barton.",
                "or": "The first Australian prime minister, who was elected in 1901, was either Sir Edmund Barton, Andrew Fisher, or Edward Deakin.",
                "pct": "I'm 52% sure that the first Australian prime minister, who was elected in 1901, was Sir Edmund Barton, but it could also have been Andrew Fisher (38% sure) or Edward Deakin (6% sure)."
        }},
        "28": {
            "original_question_idx": "333",
            "summaries": {
                "good": "It's very likely that Australia's closest French-speaking country is New Caledonia, but it could also be Papua New Guinea.",
                "bad": "Australia's closest French-speaking country is New Caledonia.",
                "or": "Australia's closest French-speaking country is either New Caledonia or Papua New Guinea.",
                "pct": "I'm 68% sure that Australia's closest French-speaking country is New Caledonia, but it could also be Papua New Guinea (20% sure)."
        }}
    }
    create_edge_case_json("majority_almostdirac", edge_case, outputs)

    # Mentioning percentages that are off (too under or overconfident)
    edge_case = {
        "0": {
            "original_question_idx": "2",
            "summaries": {
                "pct": "I'm {}% sure that Henri Becquerel won the first Nobel Prize in Physics in 1903, but it could also have been Wilhelm Conrad Röntgen in 1901 ({}% sure), or Hendrik Antoon Lorentz and Pieter Zeeman in 1902 ({}% sure)."
            },
            "percentages": [0.74, 0.1, 0.12]
        },
        "1": {
            "original_question_idx": "21",
            "summaries": {
                "pct": "I'm {}% sure that the first African American Air Force unit, the 99th Pursuit Squadron (which later became part of the Tuskegee Airmen), trained at the Selfridge Field in Michigan, but it could also have been the Tuskegee Army Air Field in Tuskegee, Alabama ({}% sure)."
            },
            "percentages": [0.52, 0.48]
        },
        "2": {
            "original_question_idx": "59",
            "summaries": {
                "pct": "I'm {}% sure that Sadio Man\u00e9 won the African Footballer of the Year award in 2014, but it could also be Sergio Agüero ({}% sure) or Serge Aurier ({}% sure)."
            },
            "percentages": [0.46, 0.32, 0.02]
        },
        "3": {
            "original_question_idx": "66",
            "summaries": {
                "pct": "I'm {}% sure that Thomas Eric Duncan died from Ebola in the United States during the 2014 West African Ebola outbreak, but it could also have been Patrick Sawyer ({}% sure)."
            },
            "percentages": [0.78, 0.06]
        },
        "4": {
            "original_question_idx": "69",
            "summaries": {
                "pct": "I'm {}% sure that the Greasers live in the east side of town in\u300aThe Outsiders\u300b, but it could be the south side of town ({}% sure)."
            },
            "percentages": [0.88, 0.08]
        },
        "5": {
            "original_question_idx": "75",
            "summaries": {
                "pct": "I'm {}% sure that the dog's name on Tom and Jerry is Jerry, but it could also be Jinx ({}% sure)."
            },
            "percentages": [0.72, 0.18]
        },
        "6": {
            "original_question_idx": "86",
            "summaries": {
                "pct": "I'm {}% sure that there are 13 countries currently part of OPEC, but it could also be 14 ({}% sure)."
            },
            "percentages": [0.9, 0.1]
        },
        "7": {
            "original_question_idx": "97",
            "summaries": {
                "pct": "I'm {}% sure that the poem \"The Woods Are Lovely, Dark and Deep\" was written by Robert Frost, but it could also have been Robert Hayden ({}% sure) or Robert Service ({}% sure)."
            },
            "percentages": [0.9, 0.02, 0.02]
        },
        "8": {
            "original_question_idx": "106",
            "summaries": {
                "pct": "I'm {}% sure that the Dallas Cowboys won their last playoff game in 2019, but it could also be 2020 ({}% sure) or 2021 ({}% sure)."
            },
            "percentages": [0.68, 0.12, 0.06]
        },
        "9": {
            "original_question_idx": "120",
            "summaries": {
                "pct": "I'm {}% sure that Mozart's Symphony No. 40 is in three movements, but it could also be four ({}% sure)."
            },
            "percentages": [0.6, 0.4]
        },
        "10": {
            "original_question_idx": "124",
            "summaries": {
                "pct": "I'm {}% sure that the fourth movie in the Divergent series, \"Divergent 2: Ascending,\" was released in 2016, but it could also be 2014 ({}% sure), 2015 ({}% sure), 2017 ({}% sure) or 2018 ({}% sure)."
            },
            "percentages": [0.48, 0.16, 0.16, 0.06, 0.06]
        },
        "11": {
            "original_question_idx": "127",
            "summaries": {
                "pct": "I'm {}% sure that Nitrogen for fertilizer comes from the Haber-Bosch process. But it could also be biological nitrogen fixation ({}% sure)."
            },
            "percentages": [0.86, 0.1]
        },
        "12": {
            "original_question_idx": "158",
            "summaries": {
                "pct": "I'm {}% sure that the first documented case of tool mark identification dates back to 1896 in France, but it could also be Spain ({}% sure) or Argentina ({}% sure)."
            },
            "percentages": [0.44, 0.18, 0.1]
        },
        "13": {
            "original_question_idx": "165",
            "summaries": {
                "pct": "I'm {}% sure that the song \"The Glory of Love\" was written by Carole King, but it could also be Lionel Richie ({}% sure), or Barry Mann and Cynthia Weil ({}% sure)."
            },
            "percentages": [0.36, 0.06, 0.04]
        },
        "14": {
            "original_question_idx": "174",
            "summaries": {
                "pct": "I'm {}% sure that this quote is from John Locke, but it could also be from George Berkeley ({}% sure) or Jonathan Swift ({}% sure)."
            },
            "percentages": [0.64, 0.18, 0.1]
        },
        "15": {
            "original_question_idx": "175",
            "summaries": {
                "pct": "I'm {}% sure that Turkish, Finnish, and Hungarian belong to the Uralic language family, but it could also be that at least Turkish belongs to the Turkic family ({}% sure)."
            },
            "percentages": [0.46, 0.52]
        },
        "16": {
            "original_question_idx": "193",
            "summaries": {
                "pct": "I'm {}% sure that Anil Kumble is the bowler that took a hattrick in Test cricket, but it could also be Jim Laker ({}%)."
            },
            "percentages": [0.46, 0.3]
        },
        "17": {
            "original_question_idx": "200",
            "summaries": {
                "pct": "I'm {}% sure that in Home Alone 2, Kevin's family goes on a cruise, but it could also be a vacation on an island ({}% sure)."
            },
            "percentages": [0.76, 0.08]
        },
        "18": {
            "original_question_idx": "226",
            "summaries": {
                "pct": "I'm {}% sure that the Oscar for Best Picture in 1976 went to \"Rocky.\" But it could also have been \"Network\" ({}% sure)."
            },
            "percentages": [0.74, 0.22]
        },
        "19": {
            "original_question_idx": "283",
            "summaries": {
                "pct": "I'm {}% sure that Wayne Rooney has scored the most goals in Premier League history, but it could also be Alan Shearer ({}% sure)."
            },
            "percentages": [0.68, 0.28]
        },
        "20": {
            "original_question_idx": "284",
            "summaries": {
                "pct": "I'm {}% sure that as of 2023, Manchester City is often considered the richest club in the Championship, but it could also be Brentford FC ({}% sure), Leeds United ({}% sure), or Aston Villa ({}% sure)."
            },
            "percentages": [0.66, 0.08, 0.06, 0.06]
        },
        "21": {
            "original_question_idx": "303",
            "summaries": {
                "pct": "I'm {}% sure that the element in group 3b (now called Group 13) and period 4 is Aluminum (Al), but it could also be Scandium (Sc) ({}% sure)."
            },
            "percentages": [0.76, 0.22]
        },
        "22": {
            "original_question_idx": "314",
            "summaries": {
                "pct": "I'm {}% sure that the actor who plays Eric Jones on Boy Meets World is Ben Savage, but it could also be Corey Feldman ({}% sure), Sean Patrick Flanery ({}% sure), or Ryan Gosling ({}% sure)."
            },
            "percentages": [0.38, 0.08, 0.04, 0.04]
        },
        "23": {
            "original_question_idx": "318",
            "summaries": {
                "pct": "I'm {}% sure that the first Australian prime minister, who was elected in 1901, was Sir Edmund Barton, but it could also have been Andrew Fisher ({}% sure) or Edward Deakin ({}% sure)."
            },
            "percentages": [0.52, 0.38, 0.06]
        },
        "24": {
            "original_question_idx": "333",
            "summaries": {
                "pct": "I'm {}% sure that Australia's closest French-speaking country is New Caledonia, but it could also be Papua New Guinea ({}% sure)."
            },
            "percentages": [0.68, 0.2]
        }
    }
    for temp in [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]:
        for key in edge_case:
            pcts = torch.Tensor(edge_case[key]["percentages"])
            pcts = torch.cat((pcts, 1. - torch.sum(pcts).unsqueeze(-1)))  # Add "Other" percentage
            pcts = torch.softmax(torch.log(pcts) / temp, dim=-1)
            pcts = pcts[:-1]  # Remove the "other" percentage
            pcts = pcts.tolist()
            pcts = [round(p * 100) for p in pcts]

            string = edge_case[key]["summaries"]["pct"]
            string = string.format(*pcts)
            edge_case[key]["summaries"][f"temp_{temp}"] = string

        # A simplified case of mentioning percentages that are off (too under or overconfident)
        edge_case = {
            "0": {
                "original_question_idx": "2",
                "summaries": {
                    "pct": "I'm {}% sure that Henri Becquerel won the first Nobel Prize in Physics in 1903, but it could also have been Wilhelm Conrad Röntgen in 1901 ({}% sure), or Hendrik Antoon Lorentz and Pieter Zeeman in 1902 ({}% sure)."
                },
                "percentages": [0.74, 0.1, 0.12]
            },
            "1": {
                "original_question_idx": "59",
                "summaries": {
                    "pct": "I'm {}% sure that Sadio Man\u00e9 won the African Footballer of the Year award in 2014, but it could also be Sergio Agüero ({}% sure) or Serge Aurier ({}% sure)."
                },
                "percentages": [0.46, 0.32, 0.02]
            },
            "2": {
                "original_question_idx": "66",
                "summaries": {
                    "pct": "I'm {}% sure that Thomas Eric Duncan died from Ebola in the United States during the 2014 West African Ebola outbreak, but it could also have been Patrick Sawyer ({}% sure)."
                },
                "percentages": [0.78, 0.06]
            },
            "3": {
                "original_question_idx": "69",
                "summaries": {
                    "pct": "I'm {}% sure that the Greasers live in the east side of town in\u300aThe Outsiders\u300b, but it could be the south side of town ({}% sure)."
                },
                "percentages": [0.88, 0.08]
            },
            "4": {
                "original_question_idx": "75",
                "summaries": {
                    "pct": "I'm {}% sure that the dog's name on Tom and Jerry is Jerry, but it could also be Jinx ({}% sure)."
                },
                "percentages": [0.72, 0.18]
            },
            "5": {
                "original_question_idx": "86",
                "summaries": {
                    "pct": "I'm {}% sure that there are 13 countries currently part of OPEC, but it could also be 14 ({}% sure)."
                },
                "percentages": [0.9, 0.1]
            },
            "6": {
                "original_question_idx": "97",
                "summaries": {
                    "pct": "I'm {}% sure that the poem \"The Woods Are Lovely, Dark and Deep\" was written by Robert Frost, but it could also have been Robert Hayden ({}% sure) or Robert Service ({}% sure)."
                },
                "percentages": [0.9, 0.02, 0.02]
            },
            "7": {
                "original_question_idx": "106",
                "summaries": {
                    "pct": "I'm {}% sure that the Dallas Cowboys won their last playoff game in 2019, but it could also be 2020 ({}% sure) or 2021 ({}% sure)."
                },
                "percentages": [0.68, 0.12, 0.06]
            },
            "8": {
                "original_question_idx": "120",
                "summaries": {
                    "pct": "I'm {}% sure that Mozart's Symphony No. 40 is in three movements, but it could also be four ({}% sure)."
                },
                "percentages": [0.6, 0.4]
            },
            "9": {
                "original_question_idx": "124",
                "summaries": {
                    "pct": "I'm {}% sure that the fourth movie in the Divergent series, \"Divergent 2: Ascending,\" was released in 2016, but it could also be 2014 ({}% sure), 2015 ({}% sure), 2017 ({}% sure) or 2018 ({}% sure)."
                },
                "percentages": [0.48, 0.16, 0.16, 0.06, 0.06]
            },
            "10": {
                "original_question_idx": "127",
                "summaries": {
                    "pct": "I'm {}% sure that Nitrogen for fertilizer comes from the Haber-Bosch process. But it could also be biological nitrogen fixation ({}% sure)."
                },
                "percentages": [0.86, 0.1]
            },
            "11": {
                "original_question_idx": "158",
                "summaries": {
                    "pct": "I'm {}% sure that the first documented case of tool mark identification dates back to 1896 in France, but it could also be Spain ({}% sure) or Argentina ({}% sure)."
                },
                "percentages": [0.44, 0.18, 0.1]
            },
            "12": {
                "original_question_idx": "174",
                "summaries": {
                    "pct": "I'm {}% sure that this quote is from John Locke, but it could also be from George Berkeley ({}% sure) or Jonathan Swift ({}% sure)."
                },
                "percentages": [0.64, 0.18, 0.1]
            },
            "13": {
                "original_question_idx": "193",
                "summaries": {
                    "pct": "I'm {}% sure that Anil Kumble is the bowler that took a hattrick in Test cricket, but it could also be Jim Laker ({}%)."
                },
                "percentages": [0.46, 0.3]
            },
            "14": {
                "original_question_idx": "200",
                "summaries": {
                    "pct": "I'm {}% sure that in Home Alone 2, Kevin's family goes on a cruise, but it could also be a vacation on an island ({}% sure)."
                },
                "percentages": [0.76, 0.08]
            },
            "15": {
                "original_question_idx": "226",
                "summaries": {
                    "pct": "I'm {}% sure that the Oscar for Best Picture in 1976 went to \"Rocky.\" But it could also have been \"Network\" ({}% sure)."
                },
                "percentages": [0.74, 0.22]
            },
            "16": {
                "original_question_idx": "283",
                "summaries": {
                    "pct": "I'm {}% sure that Wayne Rooney has scored the most goals in Premier League history, but it could also be Alan Shearer ({}% sure)."
                },
                "percentages": [0.68, 0.28]
            },
            "17": {
                "original_question_idx": "284",
                "summaries": {
                    "pct": "I'm {}% sure that as of 2023, Manchester City is often considered the richest club in the Championship, but it could also be Brentford FC ({}% sure), Leeds United ({}% sure), or Aston Villa ({}% sure)."
                },
                "percentages": [0.66, 0.08, 0.06, 0.06]
            },
            "18": {
                "original_question_idx": "303",
                "summaries": {
                    "pct": "I'm {}% sure that the element in group 3b (now called Group 13) and period 4 is Aluminum (Al), but it could also be Scandium (Sc) ({}% sure)."
                },
                "percentages": [0.76, 0.22]
            },
            "19": {
                "original_question_idx": "314",
                "summaries": {
                    "pct": "I'm {}% sure that the actor who plays Eric Jones on Boy Meets World is Ben Savage, but it could also be Corey Feldman ({}% sure), Sean Patrick Flanery ({}% sure), or Ryan Gosling ({}% sure)."
                },
                "percentages": [0.38, 0.08, 0.04, 0.04]
            },
            "20": {
                "original_question_idx": "318",
                "summaries": {
                    "pct": "I'm {}% sure that the first Australian prime minister, who was elected in 1901, was Sir Edmund Barton, but it could also have been Andrew Fisher ({}% sure) or Edward Deakin ({}% sure)."
                },
                "percentages": [0.52, 0.38, 0.06]
            },
            "21": {
                "original_question_idx": "333",
                "summaries": {
                    "pct": "I'm {}% sure that Australia's closest French-speaking country is New Caledonia, but it could also be Papua New Guinea ({}% sure)."
                },
                "percentages": [0.68, 0.2]
            }
        }
        for temp in [0.4, 1, 4]:
            for key in edge_case:
                pcts = torch.Tensor(edge_case[key]["percentages"])
                pcts = torch.cat((pcts, 1. - torch.sum(pcts).unsqueeze(-1)))  # Add "Other" percentage
                pcts = torch.softmax(torch.log(pcts) / temp, dim=-1)
                pcts = pcts[:-1]  # Remove the "other" percentage
                pcts = pcts.tolist()
                pcts = [round(p * 100) for p in pcts]

                string = edge_case[key]["summaries"]["pct"]
                string = string.format(*pcts)
                edge_case[key]["summaries"][f"temp_{temp}"] = string

    # Clean away pct summary
    for key in edge_case:
        edge_case[key]["summaries"].pop("pct")
        edge_case[key].pop("percentages")

    create_edge_case_json("pct_monotonicity", edge_case, outputs)

    # Only mentioning majority but have better wording in almost-dirac cases
    edge_case = {
        "0": {
            "original_question_idx": "2",
            "summaries": {
            "good": "It's very likely that Henri Becquerel obtained the initial Nobel Prize in Physics in 1903, but it could also have been Wilhelm Conrad Röntgen in 1901, or Hendrik Antoon Lorentz and Pieter Zeeman in 1902.",
            "bad": "Henri Becquerel won the first Nobel Prize in Physics in 1903."
        }},
        "1": {
            "original_question_idx": "14",
            "summaries": {
            "good": "It's very likely that the first declaration of human rights is connected with the French Declaration of the Rights of Man and of the Citizen, which was created in 1789 during the French Revolution. But it could also be the English Bill of Rights (1689) and the Magna Carta (1215), or Thomas Paine's Rights of Man (1791).",
            "bad": "The first declaration of human rights is associated with the French Declaration of the Rights of Man and of the Citizen, which was written in 1789 during the French Revolution.",
        }},
        "2": {
            "original_question_idx": "21",
            "summaries": {
            "good": "It's very likely that the first African American Air Force team, the 99th Pursuit Squadron (which later integrated into the Tuskegee Airmen), prepared at Selfridge Field in Michigan, but it could also have been Tuskegee Army Air Field in Tuskegee, Alabama.",
            "bad": "The first African American Air Force unit, the 99th Pursuit Squadron (which later became part of the Tuskegee Airmen), trained at Selfridge Field in Michigan."
        }},
        "3": {
            "original_question_idx": "59",
            "summaries": {
            "good": "It's very likely that Sadio Man\u00e9 was titled African Footballer of the Year in 2014, but it could also be Sergio Agüero or Serge Aurier.",
            "bad": "Sadio Man\u00e9 was named African Footballer of the Year in 2014."
        }},
        "4": {
            "original_question_idx": "66",
            "summaries": {
            "good": "Yes, it's most likely that Thomas Eric Duncan deceased from Ebola in the United States during the 2014 West African Ebola period, but it could also have been Patrick Sawyer.",
            "bad": "Yes, Thomas Eric Duncan died from Ebola in the United States during the 2014 West African Ebola outbreak."
        }},
        "5": {
            "original_question_idx": "86",
            "summaries": {
            "good": "It's very likely that there are 13 member countries of OPEC, but it could also be 14.",
            "bad": "There are 13 countries currently part of OPEC."
        }},
        "6": {
            "original_question_idx": "97",
            "summaries": {
            "good": "It's very likely that the poem \"The Woods Are Lovely, Dark and Deep\" was authored by Robert Frost, but it could also have been Robert Hayden or Robert Service.",
            "bad": "The poem \"The Woods Are Lovely, Dark and Deep\" was written by Robert Frost."
        }},
        "7": {
        "original_question_idx": "69",
        "summaries": {
            "good": "It's very likely that the Greasers reside in the east side of town in\u300aThe Outsiders\u300b, but it could be the south side of town.",
            "bad": "The Greasers live in the east side of town in\u300aThe Outsiders\u300b."
        }},
        "8": {
        "original_question_idx": "75",
        "summaries": {
            "good": "It's very likely that the dog in Tom and Jerry is called Jerry, but it could also be Jinx.",
            "bad": "The dog's name on Tom and Jerry is Jerry."
        }},
        "9": {
            "original_question_idx": "106",
            "summaries": {
                "good": "It's very likely that the Dallas Cowboys last came out victorious in their playoff game in 2019, but it could also be 2020 or 2021.",
                "bad": "The Dallas Cowboys won their last playoff game in 2019."
        }},
        "10": {
            "original_question_idx": "119",
            "summaries": {
                "good": "It's very likely that ABC broadcasts Monday Night Football, but it could also be NBC, ESPN, or FOX.",
                "bad": "ABC shows Monday Night Football."
        }},
        "11": {
            "original_question_idx": "120",
            "summaries": {
                "good": "It's very likely that Mozart's Symphony No. 40 consists of three movements, but it could also be four.",
                "bad": "Mozart's Symphony No. 40 is in three movements."
        }},
        "12": {
            "original_question_idx": "124",
            "summaries": {
                "good": "It's very likely that the fourth movie in the Divergent series, \"Divergent 2: Ascending,\" was published in 2016, but it could also be 2014, 2015, 2017 or 2018.",
                "bad": "The fourth movie in the Divergent series, \"Divergent 2: Ascending,\" was released in 2016."
        }},
        "13": {
            "original_question_idx": "127",
            "summaries": {
                "good": "It's very likely that Nitrogen for fertilizer is won from the Haber-Bosch process, which synthesizes ammonia by mixing atmospheric nitrogen with hydrogen, often obtained from natural gas. But it could also be biological nitrogen fixation by bacteria in soil and organic materials such as manure or decomposed plant matter.",
                "bad": "Nitrogen for fertilizer comes from the Haber-Bosch process, which synthesizes ammonia by combining atmospheric nitrogen with hydrogen, often sourced from natural gas."
        }},
        "14": {
            "original_question_idx": "141",
            "summaries": {
                "good": "It's most likely that the Third Five-Year Plan (1961-1966) in India was influenced by the Indo-Pak War of 1965, but it could also be the Indo-China War of 1962.",
                "bad": "The Third Five-Year Plan (1961-1966) in India was affected by the Indo-Pak War of 1965."
        }},
        "15": {
            "original_question_idx": "153",
            "summaries": {
                "good": "It's very likely that Santa is lead home by the North Star, but it could also be his reindeer Rudolph with his glowing nose.",
                "bad": "Santa is guided home by the North Star."
        }},
        "16": {
            "original_question_idx": "158",
            "summaries": {
                "good": "It's very likely that the first documented usage of tool mark identification dates back to 1896 in France, but it could also be Britain or Argentina.",
                "bad": "The first documented case of tool mark identification dates back to 1896 in France."
        }},
        "17": {
            "original_question_idx": "165",
            "summaries": {
                "good": "It's very likely that the song \"The Glory of Love\" was made by Carole King, but it could also be Lionel Richie, or Barry Mann and Cynthia Weil.",
                "bad": "The song \"The Glory of Love\" was written by Carole King."
        }},
        "18": {
            "original_question_idx": "174",
            "summaries": {
                "good": "It's very likely that John Locke gave this quote, but it could also be Jonathan Swift or George Berkeley.",
                "bad": "This quote is from John Locke."
        }},
        "19": {
            "original_question_idx": "175",
            "summaries": {
                "good": "It's very likely that Turkish, Finnish, and Hungarian are part of the Uralic languages, but it could also be that at least Turkish belongs to the Turkic family.",
                "bad": "Turkish, Finnish, and Hungarian belong to the Uralic language family."
        }},
        "20": {
            "original_question_idx": "193",
            "summaries": {
                "good": "It's very likely that Anil Kumble is the bowler that achieved a hattrick in Test cricket, but it could also be Jim Laker.",
                "bad": "Anil Kumble is the bowler that took a hattrick in Test cricket."
        }},
        "21": {
            "original_question_idx": "200",
            "summaries": {
                "good": "It's very likely that in Home Alone 2, Kevin's family departs on a cruise, but it could also be a vacation on an island.",
                "bad": "In Home Alone 2, Kevin's family goes on a cruise."
        }},
        "22": {
            "original_question_idx": "226",
            "summaries": {
                "good": "It's very likely that the Oscar for Best Picture in 1976 was given to \"Rocky.\" But it could also have been \"Network.\"",
                "bad": "The Oscar for Best Picture in 1976 went to \"Rocky.\""
        }},
        "23": {
            "original_question_idx": "283",
            "summaries": {
                "good": "It's very likely that Wayne Rooney counts the most goals in Premier League history, but it could also be Alan Shearer.",
                "bad": "Wayne Rooney has scored the most goals in Premier League history."
        }},
        "24": {
            "original_question_idx": "284",
            "summaries": {
                "good": "It's very likely that as of 2023, Manchester City is the wealthiest club in the Championship, but it could also be Leeds United, Brentford FC, or Aston Villa.",
                "bad": "As of 2023, Manchester City is the richest club in the Championship."
        }},
        "25": {
            "original_question_idx": "303",
            "summaries": {
                "good": "It's very likely that the element of group 3b (now named Group 13) and period 4 is Aluminum (Al), but it could also be Scandium (Sc).",
                "bad": "The element in group 3b (now called Group 13) and period 4 is Aluminum (Al)."
        }},
        "26": {
            "original_question_idx": "314",
            "summaries": {
                "good": "It's very likely that the actor behind Eric Jones in Boy Meets World is Ben Savage, but it could also be Corey Feldman, Sean Patrick Flanery, or Ryan Gosling.",
                "bad": "The actor who plays Eric Jones on Boy Meets World is Ben Savage."
        }},
        "27": {
            "original_question_idx": "318",
            "summaries": {
                "good": "It's very likely that the first Australian prime minister, Sir Edmund Barton, won the vote in 1901, but it could also have been Andrew Fisher or Edward Deakin.",
                "bad": "The first Australian prime minister, Sir Edmund Barton, was elected in 1901."
        }},
        "28": {
            "original_question_idx": "333",
            "summaries": {
                "good": "It's very likely that Australia's nearest French-speaking country is New Caledonia, but it could also be Papua New Guinea.",
                "bad": "Australia's closest French-speaking country is New Caledonia."
        }}
    }
    create_edge_case_json("wording_despite_majority_almostdirac", edge_case, outputs)

    # Only mentioning the most likely number when there is a range
    edge_case = {
        "0": {
            "original_question_idx": "5",
            "summaries": {
            "good": "The lowest recorded temperature on Mount Vinson lies between -67°C (-90.6°F) to -70°C (-94°F).",
            "bad": "The lowest recorded temperature on Mount Vinson is currently -68°C (-90.4°F)."
        }},
        "1": {
            "original_question_idx": "12",
            "summaries": {
            "good": "The southwest wind blows across Nigeria starts in March or April and ends in September or October.",
            "bad": "The southwest wind blows across Nigeria between April and October."
        }},
        "2": {
            "original_question_idx": "30",
            "summaries": {
            "good": "The most recent World Series to go less than seven games was somewhere between 2015 and 2020, ending in six games.",
            "bad": "The most recent World Series to go less than seven games was in 2019, ending in six games."
        }},
        "3": {
            "original_question_idx": "41",
            "summaries": {
            "good": "Lynyrd Skynyrd's \"Last of a Dying Breed\" was released somewhere between 1977 and 2016.",
            "bad": "Lynyrd Skynyrd's \"Last of a Dying Breed\" was released in 1987."
        }},
        "4": {
            "original_question_idx": "52",
            "summaries": {
            "good": "Alison Krauss's album \"Now That I've Found You\" was either released between 1997 and 2021.",
            "bad": "Alison Krauss's album \"Now That I've Found You\" was either released in 2015."
        }},
        "5": {
            "original_question_idx": "99",
            "summaries": {
            "good": "Yes, there are multiple copies of \"Sir Gawain and the Green Knight,\" with approximately four to 25 known manuscripts.",
            "bad": "Yes, there are multiple copies of \"Sir Gawain and the Green Knight,\" with approximately 25 known manuscripts."
        }},
        "6": {
            "original_question_idx": "102",
            "summaries": {
            "good": "the record for the longest motorcycle jump is between 204 and 597 feet (62.3 to 182.5 meters)",
            "bad": "The record for the longest motorcycle jump is 436 feet (133 meters)."
        }},
        "7": {
            "original_question_idx": "106",
            "summaries": {
            "good": "The Dallas Cowboys won their last playoff game between 2016 and 2023.",
            "bad": "The Dallas Cowboys won their last playoff game in 2019"
        }},
        "8": {
            "original_question_idx": "124",
            "summaries": {
            "good": "The fourth movie in the Divergent series, \"Divergent 2: Ascending,\" was released between 2014 and 2018.",
            "bad": "The fourth movie in the Divergent series, \"Divergent 2: Ascending,\" was released in 2015."
        }},
        "9": {
            "original_question_idx": "125",
            "summaries": {
            "good": "\"Republic of Doyle\" is set in the late 19th or early 20th century.",
            "bad": "\"Republic of Doyle\" is set in the 1870s."
        }},
        "10": {
            "original_question_idx": "137",
            "summaries": {
            "good": "In the United States, the average estimated cost to raise a child to age 18 is between $233,610 and over $400,000.",
            "bad": "In the United States, the average estimated cost to raise a child to age 18 is around $233,610"
        }},
        "11": {
            "original_question_idx": "185",
            "summaries": {
            "good": "The Disney Art of Animation Resort opened between 1995 and 2017.",
            "bad": "The Disney Art of Animation Resort opened in 2015."
        }},
        "12": {
            "original_question_idx": "199",
            "summaries": {
            "good": "Primary ossification centers typically appear between the 5th and 8th week of embryonic development, marking the beginning of bone formation in the developing fetus.",
            "bad": "Primary ossification centers typically appear around 8th week of embryonic development, marking the beginning of bone formation in the developing fetus."
        }},
        "13": {
            "original_question_idx": "201",
            "summaries": {
            "good": "Season 2 of Jessica Jones was released between October, 2017 and December, 2018.",
            "bad": "Season 2 of Jessica Jones was released in February, 2018."
        }},
        "14": {
            "original_question_idx": "204",
            "summaries": {
            "good": "The exact payout for players and teams can vary, but typically it ranges from $250,000 to $4 million per team.",
            "bad": "The exact payout for players and teams can vary, but typically it is about $400,000 per team."
        }},
        "15": {
            "original_question_idx": "215",
            "summaries": {
            "good": "There have been between 14 and 20 Prime Ministers of the United Kingdom.",
            "bad": "There have been 16 Prime Ministers of the United Kingdom."
        }},
        "16": {
            "original_question_idx": "218",
            "summaries": {
            "good": "The weight of a Honda Fit varies, but it generally weighs around 2,400 to 3,100 pounds (1,090 to 1,400 kg)",
            "bad": "The weight of a Honda Fit varies, but it generally weighs around 2,800 pounds (1,270 kg)"
        }},
        "17": {
            "original_question_idx": "231",
            "summaries": {
            "good": "\"When Calls the Heart\" Season 3 Episode 12 aired between December, 2015 and February, 2023.",
            "bad": "\"When Calls the Heart\" Season 3 Episode 12 aired in February, 2018."
        }},
        "18": {
            "original_question_idx": "291",
            "summaries": {
            "good": "\"Head Over Boots\" is a song by Jon Pardi, released between 2016 and 2018.",
            "bad": "\"Head Over Boots\" is a song by Jon Pardi, released in 2017."
        }},
        "19": {
            "original_question_idx": "298",
            "summaries": {
            "good": "The last game between the Philadelphia Eagles and the New England Patriots was between November, 2022 and January, 2024.",
            "bad": "The last game between the Philadelphia Eagles and the New England Patriots was in December, 2023."
        }},
        "20": {
            "original_question_idx": "311",
            "summaries": {
            "good": "Zara has approximately between 30 and 200 stores in the UK.",
            "bad": "Zara has 60 stores in the UK."
        }},
    }
    create_edge_case_json("number_ranges", edge_case, outputs)

    # Questions where the model completely hallucinates, like a uniform distribution
    edge_case = {
        "0": {
            "original_question_idx": "18",
            "summaries": {
            "good": "The owner of Reading Football Club is not known to me.",
            "bad": "The owner of Reading Football Club is Fusion Football Holdings Ltd., Fusion Capital, or Elevate Sports Holdings, LLC."
        }},
        "1": {
            "original_question_idx": "25",
            "summaries": {
            "good": 'The phrase "like a boss" gained popularity in the late 1990s and early 2000s, but I do not know where.',
            "bad": 'The phrase \"like a boss\" gained popularity in the late 1990s and early 2000s in hip-hop culture, the comedy show \"The Chris Knee Show.\", or video games.'
        }},
        "2": {
            "original_question_idx": "29",
            "summaries": {
            "good": "The first lady to be nominated as a Member of the Rajya Sabha is unknown to me.",
            "bad": "The first lady to be nominated as a Member was Sucheta Kripalani, Sushma Swaraj, or Smt. Sucheta Kriplani."
        }},
        "3": {
            "original_question_idx": "40",
            "summaries": {
            "good": "I do not know how many words the national anthem of Pakistan contains.",
            "bad": "The national anthem of Pakistan contains 39, 210, or 56 words."
        }},
        "4": {
            "original_question_idx": "41",
            "summaries": {
            "good": "I do not know when Lynyrd Skynyrd's \"Last of a Dying Breed\" was released.",
            "bad": "Lynyrd Skynyrd's \"Last of a Dying Breed\" was released in 1977, 1987, or 2014."
        }},
        "5": {
            "original_question_idx": "56",
            "summaries": {
            "good": "I do not know who the current director of the U.S. Mint is.",
            "bad": "The current director of the U.S. Mint is Michael B. White, Michael W. Wooten, or Michelle Johnson."
        }},
        "6": {
            "original_question_idx": "58",
            "summaries": {
            "good": "I do not know when Karev dies in Grey's Anatomy.",
            "bad": "Karev dies in Season 9, Episode 22, Season 11, during the episode \"An Enemy of Fate.\", or in Season 8 during the episode \"Reversion Trance.\" of Grey's Anatomy."
        }},
        "7": {
            "original_question_idx": "74",
            "summaries": {
            "good": "I do not know who plays Nikki in Need for Speed: Carbon.",
            "bad": "Nikki in Need for Speed: Carbon is played by Ellen Burroughs, Maggie Q, or Ashly Burch."
        }},
        "8": {
            "original_question_idx": "76",
            "summaries": {
            "good": "I do not know where the \"What Ifs\" music video was filmed.",
            "bad": "The \"What Ifs\" music video was filmed in Los Angeles, California, Chicago, Illinois, or in New York City."
        }},
        "9": {
            "original_question_idx": "78",
            "summaries": {
            "good": "I do not know where the Sandringham Line platforms at Flinders Street Station are.",
            "bad": "The Sandringham Line platforms at Flinders Street Station are on the north side, or Platforms 11 and 12, or in the southernmost part of the station."
        }},
        "10": {
            "original_question_idx": "81",
            "summaries": {
            "good": "I do not know who the main character in \"Harper Valley PTA\" was played by.",
            "bad": "The main character in \"Harper Valley PTA\" was played by Mary Holland, Tina Majorino, or Tammy Lauren."
        }},
        "11": {
            "original_question_idx": "85",
            "summaries": {
            "good": "I do not know where Kylo Ren's name comes from.",
            "bad": "Kylo Ren's name comes from Norse mythology, or the Polynesian word \"aki\", or from the concept of \"kaylon,\" meaning \"spear\" in Māori."
        }},
        "12": {
            "original_question_idx": "94",
            "summaries": {
            "good": "I do not know who the mother in \"The Black Stallion\" was played by.",
            "bad": "The mother in \"The Black Stallion\" was played by a horse named Pocahontas II, a horse named Cassius, or by Susan Kohlmann."
        }},
        "13": {
            "original_question_idx": "96",
            "summaries": {
            "good": "I do not know who the lyrics to \"Cant Get You Out of My Head\" were written by.",
            "bad": "The lyrics to \"Cant Get You Out of My Head\" were written by Tim Powis, Mark Wilson, and Steve Mackey, or by Tim Kilby, Mark Stoermer, and Jeff Blue, or by Tim Goldsworthy, Stephan Ewen, and Cathy Dennis."
        }},
        "14": {
            "original_question_idx": "102",
            "summaries": {
            "good": "I do not know who holds the record for the longest motorcycle jump.",
            "bad": "Kenny Brown, Kenny Bräutigam, or Kenny Frazer holds the record for the longest motorcycle jump."
        }},
        "15": {
            "original_question_idx": "133",
            "summaries": {
            "good": "I do not know who the girl in the music video for Green Day's \"21 Guns\" is.",
            "bad": "The girl in the music video for Green Day's \"21 Guns\" is Aimee Semple McPherson, Cali, or not an actual person."
        }},
        "16": {
            "original_question_idx": "151",
            "summaries": {
            "good": "I do not know when The Soul Train Music Awards usually airs.",
            "bad": "The Soul Train Music Awards usually airs in early January, late fall or early winter, or November."
        }},
        "17": {
            "original_question_idx": "182",
            "summaries": {
            "good": "I do not know who the theory of unbalanced economic growth is associated with.",
            "bad": "The theory of unbalanced economic growth is associated with Yi Chen and Ronald Findlay, Chenery and Strout, or Yi-Wen Li."
        }},
        "18": {
            "original_question_idx": "217",
            "summaries": {
            "good": "I do not know who the lyrics for \"There's a Guy Works Down the Chip Shop\" were written by.",
            "bad": "The lyrics for \"There's a Guy Works Down the Chip Shop\" were written by Terry Pratchett and Tim Field, Duncan Sheik, or Mark Lavanchy and Tim Rice."
        }},
        "19": {
            "original_question_idx": "227",
            "summaries": {
            "good": "I do not know who \"Go Tell It on the Mountain\" was popularized by.",
            "bad": "\"Go Tell It on the Mountain\" was popularized by The Five Satins, The Abyssinians, or The Gospel Choir of Jordan Av. Presbyterian Church."
        }},
        "20": {
            "original_question_idx": "233",
            "summaries": {
            "good": "I do not know who Kat Slater's two sisters in EastEnders were.",
            "bad": "Kat Slater's two sisters in EastEnders were Libby Tanner and Sally Mitchell, Tanya and Vicki Slater, or Tanya and Nicola."
        }},
        "21": {
            "original_question_idx": "244",
            "summaries": {
            "good": "I do not know who the main stars in \"Summer of '42\" include.",
            "bad": "The main stars in \"Summer of '42\" include Jacob Tremblay and Rachel Weisz, Joseph Gordon-Levitt and Anna Faris, or Jacob Tremblay and Bridget Moynahan."
        }},
    }
    create_edge_case_json("idk_cases", edge_case, outputs)

    # Being more or less verbose in a Dirac
    edge_case = {
        "0": {
            "original_question_idx": "34",
            "summaries": {
            "good": "Pyotr Ilyich Tchaikovsky composed the music for Swan Lake, The Sleeping Beauty, and The Nutcracker.",
            "bad": "Pyotr Ilyich Tchaikovsky."
        }},
        "1": {
            "original_question_idx": "47",
            "summaries": {
            "good": "Honey bees live on every continent except Antarctica. They primarily inhabit temperate and tropical regions, thriving in various climates and habitats such as forests, grasslands, agricultural areas, and urban environments.",
            "bad": "Honey bees live on every continent except Antarctica."
        }},
        "2": {
            "original_question_idx": "53",
            "summaries": {
            "good": "The first step in the evolution of the eye is thought to be the development of light-sensitive spots in simple organisms, allowing them to distinguish between light and darkness.",
            "bad": "The first step was the development of light-sensitive spots in simple organisms."
        }},
        "3": {
            "original_question_idx": "54",
            "summaries": {
            "good": "The TV show \"The Curse of Oak Island\" is primarily filmed on Oak Island, located in Mahone Bay, Nova Scotia, Canada. Some editing and additional footage may be done elsewhere, but the main location is on the island itself.",
            "bad": "\"The Curse of Oak Island\" is primarily filmed on Oak Island in Nova Scotia, Canada."
        }},
        "4": {
            "original_question_idx": "84",
            "summaries": {
            "good": "The tropical rainforest biome is characteristic of central Sub-Saharan Africa nearest the equator.",
            "bad": "Tropical rainforest biome."
        }},
        "5": {
            "original_question_idx": "91",
            "summaries": {
            "good": "Manchester United Stadium is called Old Trafford.",
            "bad": "Old Trafford."
        }},
        "6": {
            "original_question_idx": "107",
            "summaries": {
            "good": "\"Coco\" is named after the Mexican holiday Día de los Muertos (Day of the Dead). The film draws on this tradition by celebrating family, honoring ancestors, and using symbolic imagery—such as skulls—to evoke the cultural significance of the holiday. The title encapsulates the movie's themes of remembrance and the vibrant celebration of life and death.",
            "bad": "\"Coco\" is named after the Mexican holiday Día de los Muertos (Day of the Dead)."
        }},
        "7": {
            "original_question_idx": "134",
            "summaries": {
            "good": "The Ram 1500 and Ram 2500 differ primarily in size, capacity, and performance. The Ram 1500 is a light-duty truck designed for everyday use, offering a smoother ride, better fuel efficiency, and a lower towing and payload capacity, making it ideal for general hauling and personal use. In contrast, the Ram 2500 is a heavy-duty truck built for more demanding tasks, featuring a stronger frame, higher towing and payload capacity, and more powerful engine options, making it suitable for commercial use, towing heavy loads, and off-road durability. While the Ram 1500 provides better comfort and fuel efficiency, the Ram 2500 excels in power and capability, making it the preferred choice for those needing to tow trailers, RVs, or transport heavier cargo.",
            "bad": "The Ram 1500 and Ram 2500 differ primarily in size, capacity, and performance."
        }},
        "8": {
            "original_question_idx": "143",
            "summaries": {
            "good": "Atticus Finch is not born in the novel; he is a lawyer in Maycomb, Alabama.",
            "bad": "Atticus Finch is not born in the novel."
        }},
        "9": {
            "original_question_idx": "156",
            "summaries": {
            "good": "The vascular layer of the eye is the choroid.",
            "bad": "Choroid."
        }},
        "10": {
            "original_question_idx": "166",
            "summaries": {
            "good": "The brain primarily gets its energy from glucose in the blood, which is metabolized to produce ATP for cellular function. This glucose is supplied by the bloodstream and regulated by insulin.",
            "bad": "he brain primarily gets its energy from glucose in the blood."
        }},
        "11": {
            "original_question_idx": "178",
            "summaries": {
            "good": "A menstrual cup is a reusable, eco-friendly alternative to disposable pads and tampons. Instead of absorbing menstrual fluid, it collects it internally, reducing waste and providing longer-lasting protection (up to 12 hours). It is cost-effective, reduces environmental impact, and allows for comfortable period management.",
            "bad": "A menstrual cup collects menstrual fluid instead of absorbing it."
        }},
        "12": {
            "original_question_idx": "188",
            "summaries": {
            "good": "A subcutaneous injection is administered into the fatty tissue just beneath the skin, typically in areas such as the abdomen, thighs, upper arms, or buttocks.",
            "bad": "A subcutaneous injection is administered into the fatty tissue just beneath the skin."
        }},
        "13": {
            "original_question_idx": "195",
            "summaries": {
            "good": "Windows Defender is a built-in antivirus software in Windows operating systems that scans, detects, and removes viruses, malware, and other security threats. It provides real-time protection, helping to safeguard your device by continuously monitoring files and system activity. Windows Defender also includes firewall protection, cloud-based threat detection, and automatic updates to keep your system secure.",
            "bad": "Windows Defender is a built-in antivirus software in Windows operating systems."
        }},
        "14": {
            "original_question_idx": "214",
            "summaries": {
            "good": "No, a woman cannot carry twins from two different fathers. Twins from the same pregnancy typically have the same biological father, as each fertilized egg comes from the same reproductive event.",
            "bad": "No, a woman cannot carry twins from two different fathers."
        }},
        "15": {
            "original_question_idx": "216",
            "summaries": {
            "good": "Vikram Samvat calendar is officially used in Nepal. It is used alongside the Gregorian calendar. It is also used in some parts of India.",
            "bad": "Vikram Samvat calendar is officially used in Nepal."
        }},
        "16": {
            "original_question_idx": "280",
            "summaries": {
            "good": "You hear the \"lub\" and \"dub\" sounds of the heart during a stethoscope examination. These sounds occur with each heartbeat, usually audible when listening to the heart through a stethoscope.",
            "bad": "You hear the \"lub\" and \"dub\" sounds of the heart during a stethoscope examination."
        }},
        "17": {
            "original_question_idx": "292",
            "summaries": {
            "good": "No, New Orleans is not the first European town in the present-day United States. That honor goes to Saint Augustine, Florida, which was founded in 1565.",
            "bad": "No, New Orleans is not the first European town in the present-day United States."
        }},
        "18": {
            "original_question_idx": "302",
            "summaries": {
            "good": "Coastal plains of India are located along both the eastern (Bay of Bengal) and western (Arabian Sea) coasts of the country. The Eastern Coastal Plain lies along the Bay of Bengal, covering states such as West Bengal, Odisha, Andhra Pradesh, and Tamil Nadu. The Western Coastal Plain stretches along the Arabian Sea, including states like Gujarat, Maharashtra, Goa, Karnataka, and Kerala.",
            "bad": "Coastal plains of India are located along the eastern (Bay of Bengal) and western (Arabian Sea) coasts of the country."
        }},
        "19": {
            "original_question_idx": "330",
            "summaries": {
            "good": "The most common blood type in Sweden is O+, followed by A+, and then B+, and AB+.",
            "bad": "The most common blood type in Sweden is O+."
        }},
    }
    create_edge_case_json("dirac_verbosity", edge_case, outputs)

