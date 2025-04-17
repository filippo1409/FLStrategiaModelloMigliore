"""et: A Flower / sklearn app."""

from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, Parameters, EvaluateRes
from typing import Dict, List, Optional, Tuple, Callable
import pickle
import base64
from io import BytesIO
from et.task import get_model, get_model_params


class ExtraTreesFedAvg(FedAvg):
    """Strategia personalizzata per ExtraTrees."""
    
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        initial_parameters: Optional[Parameters] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, float]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, float]]] = None,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            initial_parameters=initial_parameters,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
        )
        # Memorizziamo le performance del round precedente per ciascun client
        self.client_performances = {}
        self.best_model_parameters = initial_parameters
        self.best_accuracy = 0.0
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Tuple[ClientProxy, FitRes]],
    ) -> Tuple[Optional[Parameters], Dict[str, float]]:
        """Aggregazione dei modelli ExtraTrees selezionando il modello più performante."""
        if not results:
            return None, {}
        
        # Calcoliamo il numero totale di esempi
        total_examples = sum(fit_res.num_examples for _, fit_res in results)
        print(f"Round {server_round}: {len(results)} client, {total_examples} esempi totali")
        
        # Calcoliamo i pesi per ogni client in base al numero di esempi
        weights = [fit_res.num_examples / total_examples for _, fit_res in results]
        
        client_models = []
        local_accuracies = []
        
        for i, (client, fit_res) in enumerate(results):
            params = parameters_to_ndarrays(fit_res.parameters)
            if len(params) >= 2:
                str_len = params[0][0]
                model_str = params[1].tobytes()[:str_len].decode('utf-8')
                
                try:
                    buffer = BytesIO(base64.b64decode(model_str))
                    client_model = pickle.load(buffer)
                    
                    # Estraiamo l'accuracy locale dal client, se disponibile
                    local_accuracy = fit_res.metrics.get("accuracy", 0.0)
                    
                    # In alternativa, usiamo le performance memorizzate dal round precedente
                    client_id = client.cid
                    if local_accuracy == 0.0 and client_id in self.client_performances:
                        local_accuracy = self.client_performances[client_id]
                    
                    client_models.append((client_model, weights[i], local_accuracy, client_id))
                    local_accuracies.append(local_accuracy)
                    print(f"Modello del client {i} caricato con successo, peso: {weights[i]:.4f}, accuracy: {local_accuracy:.4f}")
                except Exception as e:
                    print(f"Errore nel caricare il modello del client {i}: {e}")
        
        if not client_models:
            print("Nessun modello client valido da aggregare")
            return None, {}
        
        print(f"Aggregazione di {len(client_models)} modelli")
        
        # Inizializziamo un nuovo modello
        aggregated_model = get_model(n_estimators=10, random_state=42)

        try:
            # Prima ordiniamo per accuracy (decrescente), poi per peso (decrescente)
            sorted_clients = sorted(client_models, key=lambda x: (x[2], x[1]), reverse=True)
            
            # Informazioni di debug
            model_info = []
            for i, (model, weight, accuracy, client_id) in enumerate(sorted_clients):
                n_trees = len(model.estimators_)
                model_info.append(f"Client {i}: {n_trees} alberi, peso {weight:.4f}, accuracy {accuracy:.4f}")
            
            print("Informazioni sui modelli client (ordinati per performance):")
            for info in model_info:
                print(f"- {info}")
            
            # Usiamo il modello più performante
            best_model, _, best_accuracy, best_client_id = sorted_clients[0]
            aggregated_model.__dict__.update(best_model.__dict__)
            print(f"Selezione del client con (accuracy: {best_accuracy:.4f})")
            
        except Exception as e:
            print(f"Errore durante la selezione del modello migliore: {e}")
            # Fallback: utilizziamo il modello del primo client
            if client_models:
                best_model, _, _, best_client_id = client_models[0]
                aggregated_model.__dict__.update(best_model.__dict__)
                print(f"Fallback: utilizzato il modello del primo client disponibile (ID: {best_client_id})")
            else:
                print("Nessun modello disponibile per l'aggregazione")
                return None, {}
        return ndarrays_to_parameters(get_model_params(aggregated_model)), {}
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Tuple[ClientProxy, EvaluateRes]],
    ) -> Tuple[Optional[float], Dict[str, float]]:
        """Aggrega i risultati di valutazione e memorizza le performance per ogni client."""
        if not results:
            return None, {}
        
        # Aggregazione standard delle metriche di valutazione
        loss_aggregated = weighted_average([
            (evaluate_res.num_examples, evaluate_res.loss)
            for _, evaluate_res in results
        ])
        
        # Aggreghiamo le metriche di accuratezza
        metrics_aggregated = {"accuracy": 0.0}  # valore di default
        for client, evaluate_res in results:
            # Memorizziamo le performance di ogni client per il prossimo round
            client_id = client.cid
            if "accuracy" in evaluate_res.metrics:
                self.client_performances[client_id] = evaluate_res.metrics["accuracy"]
            
            for key, value in evaluate_res.metrics.items():
                if key not in metrics_aggregated:
                    metrics_aggregated[key] = []
                if isinstance(metrics_aggregated[key], list):
                    metrics_aggregated[key].append((evaluate_res.num_examples, value))
                else:
                    metrics_aggregated[key] = [(evaluate_res.num_examples, value)]
        
        # Calcoliamo la media pesata per ogni metrica
        for key, values in metrics_aggregated.items():
            if isinstance(values, list):
                metrics_aggregated[key] = weighted_average(values)
        
        # Aggiorniamo la migliore accuratezza globale
        accuracy = metrics_aggregated.get("accuracy", 0.0)
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
        
        print(f"Round {server_round}: Loss aggregata = {loss_aggregated}, Accuracy = {accuracy:.4f}, Miglior Accuracy = {self.best_accuracy:.4f}")
        
        return loss_aggregated, metrics_aggregated


def weighted_average(metrics: List[Tuple[int, float]]) -> float:
    """Calcola la media pesata delle metriche."""
    total_examples = sum([num_examples for num_examples, _ in metrics])
    return sum([num_examples * metric for num_examples, metric in metrics]) / total_examples


def evaluate_metrics_aggregation(eval_metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """Aggrega le metriche di valutazione dai client."""
    # Otteniamo il numero totale di esempi
    total_examples = sum([num_examples for num_examples, _ in eval_metrics])
    
    # Aggreghiamo le metriche pesate per il numero di esempi
    agg_metrics = {}
    for metric_name in eval_metrics[0][1].keys():
        weighted_sum = sum([
            num_examples * metrics[metric_name]
            for num_examples, metrics in eval_metrics
        ])
        agg_metrics[metric_name] = weighted_sum / total_examples
    
    return agg_metrics


def config_func(server_round: int) -> Dict[str, float]:
    """Restituisce la configurazione per training/evaluation."""
    return {"server_round": server_round}


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    # Create ExtraTrees Model
    n_estimators = context.run_config["n-estimators"]
    random_state = context.run_config["random-state"]
    model = get_model(n_estimators, random_state)

    # Otteniamo i parametri iniziali del modello
    initial_parameters = ndarrays_to_parameters(get_model_params(model))

    # Define strategy using ExtraTreesFedAvg strategy
    strategy = ExtraTreesFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=4,
        initial_parameters=initial_parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)
    
    print("Strategia ExtraTreesFedAvg configurata correttamente")
    print("Questa strategia utilizza un approccio semplificato che seleziona principalmente il modello migliore")

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)