# Databricks notebook source
# MAGIC %md
# MAGIC **Dominika Kaczmarska**
# MAGIC  numer indeksu: 116977
# MAGIC 
# MAGIC # Analiza czynników personalnych wpływających na depresję, niepokój i stres
# MAGIC 
# MAGIC ### 1. Opis problemu
# MAGIC Depresja, niepokój i stres to stany, które dotykają wiele osób. W celu określenia skali tych stanów są stworzone różne testy, a jednym z nich jest Depression Anxiety Stress Scales (DASS). W wersji online tego testu oprócz pytań określających stopień zaburzeń, pojawiły się pytania personalne takie jak wiek, płeć, orientacja. 
# MAGIC 
# MAGIC Analizowany zbiór zawiera dane osób, które wypełniały test w latach 2017-2019 oraz wyraziły zgodę na wykorzystanie wyników do badań.
# MAGIC 
# MAGIC Analiza tego zbioru pozwoli określić charakterystykę tych zaburzeń.
# MAGIC 
# MAGIC 
# MAGIC _Źródło danych:_ https://www.kaggle.com/yamqwe/depression-anxiety-stress-scales

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Przygotowanie danych

# COMMAND ----------

# MAGIC %md ###### Wczytanie danych z pliku CSV

# COMMAND ----------

d = spark.read\
  .format("csv")\
  .options(
    inferSchema="true",
    header="true",
    delimiter="	")\
  .load("/FileStore/tables/data.csv")

# COMMAND ----------

# MAGIC %md Badanie zawiera 42 pytania dotyczące zaburzeń, na które były do wyboru cztery odpowiedzi numeryczne:
# MAGIC * 1 - zupełnie mnie nie dotyczy
# MAGIC * 2 - dotyczy mnie w pewnym stopniu, albo dotyczy tylko czasami
# MAGIC * 3 - dotyczy mnie w znaczącym stopniu, albo dotyczy mnie często
# MAGIC * 4 - zdecydowanie mnie dotyczy, albo dotyczy mnie prawie zawsze
# MAGIC 
# MAGIC Każde pytanie przypisane jest do jednego z trzech zaburzeń - depresji, stresu lub niepokojum - a zsumowanie odpowiedzi dla danego zaburzenia pozwala określić jego stopień.

# COMMAND ----------

# MAGIC %md ###### Zsumowanie odpowiedzi na pytania w celu określenia stopnia zaburzenia

# COMMAND ----------

d2 = d.withColumn("Depresja", d.Q3A + d.Q5A + d.Q10A + d.Q13A + d.Q16A + d.Q17A + d.Q21A + d.Q24A + d.Q26A + d.Q31A + d.Q34A + d.Q37A + d.Q38A + d.Q42A)\
    .withColumn("Niepokój", d.Q2A + d.Q4A + d.Q7A + d.Q9A + d.Q15A + d.Q19A + d.Q20A + d.Q23A + d.Q25A + d.Q28A + d.Q30A + d.Q36A + d.Q40A + d.Q41A)\
    .withColumn("Stres", d.Q1A + d.Q6A + d.Q8A + d.Q11A + d.Q12A + d.Q14A + d.Q18A + d.Q22A + d.Q27A + d.Q29A + d.Q32A + d.Q33A + d.Q35A + d.Q39A)

# COMMAND ----------

# MAGIC %md ###### Wybór interesujących nas kolumn

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Kolumny, które będą przez nas używane w analizie to:
# MAGIC * Depresja 
# MAGIC * Niepokój 
# MAGIC * Stres 
# MAGIC * Płeć 
# MAGIC * Wiek
# MAGIC * Wyznanie
# MAGIC * Orientacja
# MAGIC * Stan_cywilny
# MAGIC * Wykształcenie

# COMMAND ----------

d3 = d2.select("Depresja", "Niepokój", "Stres", "gender", "age", "religion", "orientation", "married", "education")\
        .toDF("Depresja", "Niepokój", "Stres", "Płeć", "Wiek", "Wyznanie", "Orientacja", "Stan_cywilny", "Wykształcenie")

# COMMAND ----------

d3.display()

# COMMAND ----------

# MAGIC %md  W celu łatwiejszej analizy danych zamienimy zmienne liczbowe przedstawiające stopień zaburzeń na zmienne kategoryczne (przedziały dla danych kategorii są zdefiniowane w teście z którego pochodzą dane) oraz zmienne Płeć, Religia, Orientacja, Rada, Stan_cywilny, Wykształcenie na zmienne tekstowe (zgodnia z kluczem podanym w dokumentacji źródła danych).

# COMMAND ----------

def Depresja(x):
    if x <= 24:
        return "Brak"
    elif x >= 25 and x <= 27:
        return "Lekki"
    elif x >= 28 and x <= 34:
        return "Średni"
    elif x >= 35 and x <= 41:
        return "Poważny"
    else:
        return "Bardzo poważny"

# COMMAND ----------

def Niepokój(x):
    if x <= 21:
        return "Brak"
    elif x >= 22 and x <= 23:
        return "Lekki"
    elif x >= 24 and x <= 28:
        return "Średni"
    elif x >= 29 and x <= 33:
        return "Poważny"
    else:
        return "Bardzo poważny"

# COMMAND ----------

def Stres(x):
    if x <= 28:
        return "Brak"
    elif x >= 29 and x <= 32:
        return "Lekki"
    elif x >= 33 and x <= 39:
        return "Średni"
    elif x >= 41 and x <= 47:
        return "Poważny"
    else:
        return "Bardzo poważny"

# COMMAND ----------

from pyspark.sql.types import *

# COMMAND ----------

udf_Depresja = udf(Depresja, StringType())
udf_Niepokój = udf(Niepokój, StringType())
udf_Stres = udf(Stres, StringType())

# COMMAND ----------

d4 = d3.withColumn("Depresja", udf_Depresja(d3.Depresja))
d4 = d4.withColumn("Niepokój", udf_Niepokój(d3.Niepokój))
d4 = d4.withColumn("Stres", udf_Stres(d3.Stres))

# COMMAND ----------

import pyspark.sql.functions as f

# COMMAND ----------

d4 = d4.withColumn("Płeć",
                  f.when(f.col("Płeć") == 1, "Mężczyzna")\
                   .when(f.col("Płeć") == 2, "Kobieta")\
                   .when(f.col("Płeć") == 3, "Inna")
                  )

# COMMAND ----------

d4 = d4.withColumn("Wyznanie",
                  f.when(f.col("Wyznanie") == 1, "Agnostyk")\
                   .when(f.col("Wyznanie") == 2, "Ateista")\
                   .when(f.col("Wyznanie") == 3, "Buddysta")\
                   .when(f.col("Wyznanie") == 4, "Chrześcijanin (Katolik)")\
                   .when(f.col("Wyznanie") == 5, "Chrześcijanin (Mormon)")\
                   .when(f.col("Wyznanie") == 6, "Chrześcijanin (Protestant)")\
                   .when(f.col("Wyznanie") == 7, "Chrześcijanin (Inny odłam)")\
                   .when(f.col("Wyznanie") == 8, "Hindus")\
                   .when(f.col("Wyznanie") == 9, "Żyd")\
                   .when(f.col("Wyznanie") == 10, "Muzułmanin")\
                   .when(f.col("Wyznanie") == 11, "Sikh")\
                   .when(f.col("Wyznanie") == 12, "Inne")
                  )

# COMMAND ----------

d4 = d4.withColumn("Orientacja",
                  f.when(f.col("Orientacja") == 1, "Heteroseksualizm")\
                   .when(f.col("Orientacja") == 2, "Biseksualizm")\
                   .when(f.col("Orientacja") == 3, "Homoseksualizm")\
                   .when(f.col("Orientacja") == 4, "Aseksualizm")\
                   .when(f.col("Orientacja") == 5, "Inna")
                  )

# COMMAND ----------

d4 = d4.withColumn("Stan_cywilny",
                  f.when(f.col("Stan_cywilny") == 1, "Panna/Kawaler")\
                   .when(f.col("Stan_cywilny") == 2, "Zamężna/Żonaty")\
                   .when(f.col("Stan_cywilny") == 3, "Rozwiedziona/Rozwiedziony")
                  )

# COMMAND ----------

d4 = d4.withColumn("Wykształcenie",
                f.when(f.col("Wykształcenie") == 1, "Podstawowe")\
                 .when(f.col("Wykształcenie") == 2, "Średnie")\
                 .when(f.col("Wykształcenie") == 3, "Wyższe 1 stopnia")\
                 .when(f.col("Wykształcenie") == 4, "Wyższe 2 stopnia")
                )

# COMMAND ----------

d4.display()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 3. Analiza eksploracyjna danych

# COMMAND ----------

order = spark.createDataFrame([[1, "Brak"],
                               [2, "Lekki"],
                               [3, "Średni"],
                               [4, "Poważny"],
                               [5, "Bardzo poważny"]],
                              schema=["kolejność", "Poziom"])

# COMMAND ----------

order_edu = spark.createDataFrame([[1, "Podstawowe"],
                               [2, "Średnie"],
                               [3, "Wyższe 1 stopnia"],
                               [4, "Wyższe 2 stopnia"]],
                              schema=["kolejność_wykształcenie", "wykszt"])

# COMMAND ----------

d5 = d4.groupBy("Depresja").count().withColumn("Zaburzenie", f.lit("Depresja"))\
        .union(d4.groupBy("Niepokój").count().withColumn("Zaburzenie", f.lit("Niepokój")))\
        .union(d4.groupBy("Stres").count().withColumn("Zaburzenie", f.lit("Stres")))\
    .join(order, on = (f.col("Poziom")==f.col("Depresja")), how = "left")\
    .orderBy(f.col("kolejność"))

# COMMAND ----------

d5.display()

# COMMAND ----------

d5.select(f.col("Zaburzenie"), f.col("Poziom"), f.col("count"))\
    .join(d5.groupBy(f.col("Zaburzenie")).sum("count"), on = "Zaburzenie", how = "left")\
    .withColumn("Procent", 100*f.col("count")/f.col("sum(count)"))\
    .filter("Poziom like 'Bardzo poważny' OR Poziom like 'Poważny'")\
    .groupBy(f.col("Zaburzenie"))\
    .sum("Procent")\
    .withColumn("Procent", f.bround(f.col("sum(Procent)"), 0))\
    .select(f.col("Zaburzenie"), f.col("Procent"))\
    .display()

# COMMAND ----------

# MAGIC %md Depresja i niepokój są częściej występującym zaburzeniem, niż stres - występują o około 15% częściej.

# COMMAND ----------

# MAGIC %md #### 3.1. Niepokój

# COMMAND ----------

d6 = d4.groupBy("Niepokój", "Orientacja").count()\
        .join(d4.groupBy("Orientacja").count().withColumnRenamed("count", "calosc"), on = "Orientacja", how = "left")\
    .filter(d4.Orientacja != "null")\
    .withColumn("Procent", f.bround(100*f.col("count")/f.col("calosc"),2))\
    .join(order, on = (f.col("Poziom")==f.col("Niepokój")), how = "left")\
    .orderBy(f.col("kolejność"))

# COMMAND ----------

d6.display()

# COMMAND ----------

d6.filter("Poziom like 'Poważny' OR Poziom like 'Bardzo poważny'")\
.select(f.col("Orientacja"), f.col("Poziom"), f.col("Procent"))\
.groupBy(f.col("Orientacja"))\
.sum("Procent")\
.withColumn("Procent", f.bround("sum(Procent)", 0))\
.select(f.col("Orientacja"), f.col("Procent"))\
.orderBy("Procent")\
.display()

# COMMAND ----------

# MAGIC %md Najmniej narażoną na odczuwanie niepokoju grupą osób są osoby heteroseksualne - niepokój odczuwa o ponad 10% mniej heteroseksualnych osób, niż osób innej orientacji.

# COMMAND ----------

d4.groupBy("Niepokój", "Płeć").count()\
.filter(d4.Płeć != "null")\
.join(order, on = (f.col("Poziom")==f.col("Niepokój")), how = "left")\
.orderBy(f.col("kolejność"))\
.display()

# COMMAND ----------

# MAGIC %md Grupą najbardziej narażoną na odczuwanie niepokoju są osoby, które identyfikują swoją płeć jako "Inna", grupą najmniej narażoną są mężczyźni:
# MAGIC * około 25% mniej mężczyzn odczuwa niepokój niż osoby, które nie identyfikują się jako kobieta/mężczyzna
# MAGIC * około 15 % więcej kobiet odczuwa niepokój niż mężczyzn

# COMMAND ----------

d4.groupBy("Niepokój", "Wykształcenie").count()\
.filter(d4.Wykształcenie != "null")\
.join(order, on = (f.col("Poziom")==f.col("Niepokój")), how = "left")\
.join(order_edu, on = (f.col("Wykształcenie")==f.col("wykszt")), how = "left")\
.orderBy(f.col("kolejność"), f.col("kolejność_wykształcenie"))\
.display()

# COMMAND ----------

# MAGIC %md Im wyższe wykształcenie, tym mniej osób odczuwa niepokój - między osobami posiadającymi wykształcenie podstawowe, a wykształcenie wyższe 2 stopnia różnica wynosi aż 27% (biorąc pod uwagę poziom niepokoju poważny oraz bardzo poważny).

# COMMAND ----------

d4.groupBy("Niepokój", "Stan_cywilny").count()\
.filter(d4.Stan_cywilny != "null")\
.join(order, on = (f.col("Poziom")==f.col("Niepokój")), how = "left")\
.orderBy(f.col("kolejność"))\
.display()

# COMMAND ----------

# MAGIC %md Grupą najbardziej narażoną na niepokój są panny/kawalerowie, a najmniej narażone są osoby będące w związku małżeńskim. Biorąc pod uwagę poziom niepokoju poważny i bardzo poważny:
# MAGIC * około 15% mniej rozwodników odczuwa niepokój, niż osoby które nigdy nie były w związku małżeńskim
# MAGIC * około 25% mniej osób będących w związku małżeńskim odczuwa niepokój, w porównaiu do osoby które nigdy w związku małżeńskim nie były

# COMMAND ----------

d4.createOrReplaceTempView("tmp_d4")

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT `Niepokój`, tab1.Wiek, round(100*count/calosc,2) AS Procent 
# MAGIC FROM ((SELECT `Niepokój`, Wiek, count(*) AS count FROM tmp_d4 GROUP BY `Niepokój`, Wiek) tab1 
# MAGIC         JOIN (SELECT Wiek, count(*) AS calosc FROM tmp_d4 GROUP BY Wiek) tab2 ON tab1.Wiek = tab2.Wiek)
# MAGIC WHERE calosc > 1 and (`Niepokój` like 'Poważny' OR `Niepokój` like 'Bardzo poważny')
# MAGIC SORT BY Wiek

# COMMAND ----------

# MAGIC %md Na niepokój zdecydowanie bardziej są narażone osoby młode. U osób w wieku 13 do 35 lat następuje stopniowy spadek odczuwania niepokoju. Pomiędzy osobami w wieku 13, a 35 lat różnica wynosi 40%. Później poziom niepokoju utrzymuje się na zbliżonym poziomie.

# COMMAND ----------

# MAGIC %md #### 3.2. Depresja

# COMMAND ----------

d4.groupBy("Depresja", "Płeć").count()\
.filter(d4.Płeć != "null")\
.join(order, on = (f.col("Poziom")==f.col("Depresja")), how = "left")\
.orderBy(f.col("kolejność"))\
.display()

# COMMAND ----------

# MAGIC %md Grupą osób które są najbardziej narażone na odczuwanie depresji są osoby, które nie identyfikują się ani jako kobieta ani jako mężczyzna - jest to około 20% więcej osób w porównaniu tych, które określają swoją płeć jako męska lub żeńska.

# COMMAND ----------

d7 = d4.groupBy("Depresja", "Wyznanie").count()\
        .join(d4.groupBy("Wyznanie").count().withColumnRenamed("count", "calosc"), on = "Wyznanie", how = "left")\
    .filter(d4.Wyznanie != "null")\
    .withColumn("Procent", f.bround(100*f.col("count")/f.col("calosc"),2))\
    .join(order, on = (f.col("Poziom")==f.col("Depresja")), how = "left")\
    .orderBy(f.col("kolejność"))

# COMMAND ----------

d7.display()

# COMMAND ----------

d7.filter("Poziom like 'Poważny' OR Poziom like 'Bardzo poważny'")\
.select(f.col("Wyznanie"), f.col("Poziom"), f.col("Procent"))\
.groupBy(f.col("Wyznanie"))\
.sum("Procent")\
.withColumn("Procent", f.bround("sum(Procent)", 0))\
.select(f.col("Wyznanie"), f.col("Procent"))\
.orderBy("Procent")\
.display()

# COMMAND ----------

# MAGIC %md Najbardziej narażeni na depresję są ateiści oraz agnostycy, a najmniej buddyści - różnica między tymi skrajnymi wartościami wynosi 18%.

# COMMAND ----------

d4.groupBy("Depresja", "Stan_cywilny").count()\
.filter(d4.Stan_cywilny != "null")\
.join(order, on = (f.col("Poziom")==f.col("Depresja")), how = "left")\
.orderBy(f.col("kolejność"))\
.display()

# COMMAND ----------

# MAGIC %md Najmniej narażone na depresje są osoby, które są w związku małżeńskim:
# MAGIC * około 20% mniej osób w związku małżeńskim jest narażonych na depresję, niż osób które nigdy w takim związku nie były
# MAGIC * około 15% więcej osób po rozwodzie jest narażonych na depresję, niż osób w związku małżeńskim

# COMMAND ----------

# MAGIC %md #### 3.3. Stres

# COMMAND ----------

d4.groupBy("Stres", "Wykształcenie").count()\
.filter(d4.Wykształcenie != "null")\
.join(order, on = (f.col("Poziom")==f.col("Stres")), how = "left")\
.join(order_edu, on = (f.col("Wykształcenie")==f.col("wykszt")), how = "left")\
.orderBy(f.col("kolejność"), f.col("kolejność_wykształcenie"))\
.display()

# COMMAND ----------

# MAGIC %md Im wyższe wykształcenie, tym mniej osób odczuwa stres - różnica między osobami posiadającymi wykształcenie podstawowe, a wykształcenie 2 stopnia wynosi 20%.

# COMMAND ----------

d8 = d4.groupBy("Stres", "Wyznanie").count()\
        .join(d4.groupBy("Wyznanie").count().withColumnRenamed("count", "calosc"), on = "Wyznanie", how = "left")\
    .filter(d4.Wyznanie != "null")\
    .withColumn("Procent", f.bround(100*f.col("count")/f.col("calosc"),2))\
    .join(order, on = (f.col("Poziom")==f.col("Stres")), how = "left")\
    .orderBy(f.col("kolejność"))

# COMMAND ----------

d8.display()

# COMMAND ----------

d8.filter("Poziom like 'Poważny' OR Poziom like 'Bardzo poważny'")\
.select(f.col("Wyznanie"), f.col("Poziom"), f.col("Procent"))\
.groupBy(f.col("Wyznanie"))\
.sum("Procent")\
.withColumn("Procent", f.bround("sum(Procent)", 0))\
.select(f.col("Wyznanie"), f.col("Procent"))\
.orderBy("Procent")\
.display()

# COMMAND ----------

# MAGIC %md Najbardziej narażeni na stres są mormoni - stres odczuwa ponad 10% więcej osób, niż osób z pozostałych wyznań.
# MAGIC Najmniej narażeni na stres są buddyści - różnica między buddystami i mormonami wynosi aż 22%.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Podsumowanie
# MAGIC 
# MAGIC Dokonując analizy wyników testu, można wyodrębnić kilka grup najbardziej/najmniej podatnych na takie zaburzenia jak depresja, stres i niepokój. W podziale na kategorie:
# MAGIC * **Płeć**: Na niepokój oraz na depresję najbardziej są narażone osoby, które nie identyfikują swojej płci jako kobieta ani jako mężczyzna. Mężczyźni są mniej podatni na niepokój niż kobiety.
# MAGIC * **Wiek**: Osoby nastoletnie zdecydowanie bardziej odczuwają niepokój, niż osoby będą powyżej 30 roku życia.
# MAGIC * **Wyznanie**: Buddiści są grupą, która w najmniejszym stopniu odczuwa depresję oraz stres. Najczęściej depresję odczuwają ateiści oraz agnostycy. Natomiast stres najczęściej odczuwają mormoni.
# MAGIC * **Orientacja**: Najbardziej podatni na niepokój są biseksualiści. Najmniej natomiast heteroseksualiści.
# MAGIC * **Stan cywilny**: Najmniejszą grupą osób odczuwająca niepokój oraz depresję są ludzie, którzy są w związku małżeńskim.
# MAGIC * **Wykształcenie**: Im wyższe wykształcenie, tym mniej ludzi odczuwa niepokój oraz stres.
