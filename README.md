# TFM_Earthquake_forecasting_based_on_Transformers
En el presente Trabajo Fin de Máster se ha diseñado, desarrollado y evaluado una metodología de pronóstico de terremotos basada en redes Transformer. 
Paralelamente, también se han desarrollado y evaluado las metodologías LightGBM y Random Forest con el fin de comparar los resultados entre las tres técnicas.
El ámbito de aplicación ha sido la provincia de Alicante, con una sismicidad histórica moderada–alta. La provincia de Alicante está dividida en cuatro zonas sismogénicas diferentes, las cuales han sido analizadas de forma independiente en todo el proceso llevado a cabo.
En el presente trabajo se han planteado hasta 113 características sísmicas diferentes, las cuales han sido analizadas mediante los algoritmos LightGBM y SHAP en busca de determinar los parámetros más influyentes en el pronóstico de los terremotos. De ese modo se han seleccionado los 21–23 parámetros más importantes dependiendo de la zona sismogénica analizada.  
En las pruebas realizadas para cada zona sismogénica también se ha analizado la influencia de la clasificación del vector objetivo y se ha estimado la mejor configuración para la red Transformer. 
La metodología propuesta estima si en los próximos 30 días va a ocurrir un terremoto de cierta magnitud. La evaluación de las metodologías LightGBM, Random Forest y la red Transformer proporciona valores muy parecidos, especialmente para la clase correspondiente a los eventos de mayor magnitud (clase 3). En este caso, los resultados obtenidos para esta clase superan el 90% de precisión (en test) para todas las metodologías y zonas sismogénicas.
El rendimiento obtenido por la metodología propuesta, basada en la red Transformer, es completamente comparable con el rendimiento de las técnicas LightGBM y Random Forest, y dobla el valor del rendimiento de este tipo de redes en otros trabajos previos.


Palabras clave: Transformer, Random Forest, LightGBM, pronóstico de terremotos, características sísmicas.
